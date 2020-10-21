import numpy as np
from typing import Union, Callable, List
from threading import Event
from ..agents.agent import Agent
from ..problems import ConvexifiedMILP, ConstraintCoupledMILP, MixedIntegerLinearProblem
from .primal_decomp import PrimalDecomposition
from ..functions import Variable
from . import MaxConsensus


class PrimalDecompositionMILP(PrimalDecomposition):
    """Distributed primal decomposition for MILPs.
    """

    # TODO choose ref
    def __init__(self, agent: Agent, initial_condition: np.ndarray, enable_log: bool = False,
        restriction: np.ndarray = None, finite_time_add_restriction: float = 0):

        super().__init__(agent, initial_condition, enable_log)

        # check input
        if not isinstance(agent.problem, ConstraintCoupledMILP):
            raise TypeError("The agent must be equipped with a ConstraintCoupledMILP")

        # initialize some variables and local MILP solver
        self.fast_mode = True
        param = agent.problem.coupling_function.get_parameters()
        self.A = param[0].T
        self.cutting_planes = None
        self.restriction = restriction
        self.ft_restriction = finite_time_add_restriction
        self.integer_vars = agent.problem.integer_vars
        self.binary_vars = agent.problem.binary_vars
        self.milp = ConvexifiedMILP(objective_function = agent.problem.objective_function,
            y = np.zeros((self.S, 1)), A = self.A, constraints = agent.problem.constraints,
            integer_vars = self.integer_vars, binary_vars = self.binary_vars)
        
        # further parameters of local solver
        self.max_local_iterations = 1000
        self.local_threshold_convergence = 1e-8
        self.viol_additional_slack = 1e-5 # additional slack in case of violation

    def run(self, n_agents: int, iterations: int = 1000, stepsize: Union[float, Callable] = 0.1, M: float = 1000.0,
        verbose: bool=False, callback_iter: Callable=None, fast_mode: bool=True,
        max_cutting_planes: int=1000, milp_solver: str=None,
        extra_allocation: np.ndarray = None,
        use_runavg: bool=False, # use running average of y in local MILP?
        runavg_start_iter: int=0, event: Event=None,
        max_consensus_iterations: int = None,
        max_consensus_graph_diam: int = None, **kwargs) -> np.ndarray:
        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 1000.
            stepsize: If a float is given as input, the stepsize is constant. 
                      If a function is given, it must take an iteration k as input and output the corresponding stepsize. Defaults to 0.1.
            M: Value of the parameter :math:`M`. Defaults to 1000.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
            callback_iter: callback function to be called at the end of each iteration. Must take an iteration k as input. Defaults to None.
            fast_mode: If True, a mixed-integer solution is computed at each iteration, otherwise only at the last iteration. Defaults to True.
            max_cutting_planes: maximum number of cutting planes for local solver. Defaults to 1000.
            milp_solver: MILP solver to use. Defaults to None (use default solver).

        Raises:
            TypeError: The number of iterations must be an int
            TypeError: The stepsize must be a float or a callable
            TypeError: The parameter M must be a float

        Returns:
            return a tuple (x, y) with the sequence of primal solutions and allocation estimates if enable_log=True.
            If fast_mode=True, the primal solutions are those of the convexified problem, otherwise they are the mixed-integer estimates.
        """

        # store parameters
        self.milp_solver = milp_solver
        self.max_cutting_planes = max_cutting_planes
        self.fast_mode = fast_mode
        self.extra_allocation = extra_allocation if extra_allocation is not None else np.zeros((self.S, 1))
        self.use_runavg = use_runavg

        # compute and apply restriction
        self.compute_restriction(max_consensus_iterations, max_consensus_graph_diam)
        self.y = self.y0 - (self.restriction + self.ft_restriction)/n_agents

        if verbose and self.agent.id == 0:
            total_restriction = np.array(self.restriction + self.ft_restriction)
            print('Overall applied restriction: {}'.format(total_restriction.flatten()))

        # run primal decomposition algorithm on convexified problem
        result = super().run(iterations, stepsize, M, verbose, callback_iter, compute_runavg=use_runavg, runavg_start_iter=runavg_start_iter, event=event, **kwargs)

        # compute mixed-integer solution (if not already available)
        if fast_mode and (event is None or not event.is_set()): # skip if optimization has been interrupted
            self.x = self.mixed_integer_solution()
        
        return result
    
    def iterate_run(self, stepsize: float, M: float, update_runavg, event: Event, **kwargs):
        """Run a single iterate of the algorithm
        """

        # solve local problem
        out = self.milp.solve(M=M, milp_solver=self.milp_solver, return_only_solution=False,
                y=self.y, cutting_planes=self.cutting_planes, max_iterations=self.max_local_iterations,
                threshold_convergence=self.local_threshold_convergence)

        # save data
        x = out[0][0]
        mu = out[2]
        self.cutting_planes = out[3]

        # exchange dual variables with neighbors
        data = self.agent.neighbors_exchange(mu, event=event)
        mu_neigh = [data[idx] for idx in data]

        if event is None or not event.is_set():
            self._update_local_solution(x, mu, mu_neigh, stepsize, update_runavg, **kwargs)

            # solve local MILP if requested
            if not self.fast_mode:
                self.x = self.mixed_integer_solution()
    
    def compute_restriction(self, iterations, graph_diam):
        # check if restriction is already available
        if self.restriction is not None:
            return
        
        # compute lower and upper bounds of allocations
        LB = np.zeros((self.S, 1))
        UB = np.zeros((self.S, 1))

        for s in range(self.S):
            x = Variable(self.x_shape[0])
            obj_func = self.A[s, :][:, None] @ x

            # solve s-th lower bound problem
            milp = MixedIntegerLinearProblem(obj_func, self.agent.problem.constraints, self.integer_vars, self.binary_vars)
            x_opt = milp.solve() if self.milp_solver is None else milp.solve(solver=self.milp_solver)
            
            LB[s] = obj_func.eval(x_opt)

            # solve s-th upper bound problem
            milp.objective_function = -obj_func
            x_opt = milp.solve() if self.milp_solver is None else milp.solve(solver=self.milp_solver)
            
            UB[s] = obj_func.eval(x_opt)

        # prepare minimum resource MILP
        A = np.hstack((np.zeros((self.S, self.x_shape[0])), np.ones((self.S,1)))).transpose() # matrix for \rho \1
        A_rho = np.hstack((np.zeros((1, self.x_shape[0])), [[1]])).transpose() # matrix selecting only rho variable

        z = Variable(self.x_shape[0] + 1) # symbolic variable [x, rho]
        rho = A_rho @ z # symbolic variable (only rho)

        constraints = [self.coupling_function <= LB + A @ z]
        constraints.extend(self.local_constraints)

        # solve MILP to compute vector using minimal resources
        milp = MixedIntegerLinearProblem(rho, constraints, self.integer_vars, self.binary_vars)
        z_opt = milp.solve() if self.milp_solver is None else milp.solve(solver=self.milp_solver)

        # compute worst-case local violation (finally!)
        min_resource = rho.eval(z_opt) * np.ones((self.S, 1))
        restriction_loc = np.minimum(UB - LB, min_resource)

        ######################

        # run max-consensus algorithm
        algorithm = MaxConsensus(self.agent, restriction_loc, graph_diam)

        if iterations is not None:
            algorithm.run(iterations=iterations)
        else:
            algorithm.run()

        # save result
        self.restriction = self.S * algorithm.get_result()
    
    def mixed_integer_solution(self):

        allocation = self.y_avg if self.use_runavg else self.y

        y_viol = self.compute_violating_y(allocation)

        # prepare final MILP
        x = Variable(self.x_shape[0])
        constraints = [self.A.T@x <= y_viol + self.extra_allocation]
        constraints.extend(self.agent.problem.constraints)

        # solve final MILP
        milp_final = MixedIntegerLinearProblem(self.agent.problem.objective_function, constraints,
            self.integer_vars, self.binary_vars)
        x_final = milp_final.solve() if self.milp_solver is None else milp_final.solve(solver=self.milp_solver)

        return x_final
    
    def compute_violating_y(self, allocation):

        A = np.hstack((np.zeros((self.S, self.x_shape[0])), np.ones((self.S,1)))).transpose() # matrix for \rho \1
        A_rho = np.hstack((np.zeros((1, self.x_shape[0])), [[1]])).transpose() # matrix selecting only rho variable

        z = Variable(self.x_shape[0] + 1) # symbolic variable [x, rho]
        rho = A_rho @ z # symbolic variable (only rho)
        alloc_constr = self.coupling_function <= allocation + self.extra_allocation + A @ z
        constraints = [alloc_constr, rho >= 0]
        constraints.extend(self.local_constraints)

        # solve MILP to compute violation
        milp_viol = MixedIntegerLinearProblem(rho, constraints, self.integer_vars, self.binary_vars)
        z_opt = milp_viol.solve() if self.milp_solver is None else milp_viol.solve(solver=self.milp_solver)

        viol = rho.eval(z_opt) # evaluate rho at optimal solution

        # vectorize violation
        violation = (viol + self.viol_additional_slack) * np.ones((self.S, 1))

        return allocation + violation
