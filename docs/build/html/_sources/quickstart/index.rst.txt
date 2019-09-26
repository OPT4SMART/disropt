.. _quickstart:

Quick start
===================================

For the installation of the package, refer to the :ref:`install` section.

To run an algorithm, it suffices to create an instance of the corresponding class and then call the method :code:`run()`.
The class constructor requires an instance of the :ref:`Agent <agent>` class, which must contain the local
information available to the agent to run the algorithm.

Example with the consensus algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example, to run the :ref:`Consensus <alg_consensus>` algorithm, first create an instance of the Agent
class with the graph information::

    agent = Agent(in_neighbors, out_neighbors, in_weights)

where the variables :code:`in_neighbors`, :code:`out_neighbors` and :code:`in_weights` are previously
initialized lists. Then, create an instance of the Consensus class with the agent's initial condition
and call the method :code:`run()`::

    algorithm = Consensus(agent=agent, initial_condition=x0)
    algorithm.run(iterations=100)


The method :code:`get_result()` can be called to get the output of the algorithm::

    print("Output of agent {}: {}".format(agent.id, algorithm.get_result()))

All the code showed so far is python code and must be enclosed in a script file. To actually run
the code with MPI (which is the default :ref:`Communicator <communicators>`), run on a terminal:

.. code-block:: bash

    mpirun -np 8 python script.py

where in this case the script file :code:`script.py` is executed over 8 processors.

Example with distributed optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For distributed optimization algorithms, the workflow is almost the same, except that the Agent class
must be equipped with the problem data that is locally available to the agent. The problem data should
be passed as an instance of the :ref:`Problem <problem>` class (or one of its children) *before* creating
the instance of the algorithm class.

For example, to run the :ref:`Distributed subgradient <alg_subgradient>` algorithm, the cost function
must be passed to the instance of the Agent class after its initialization::

    problem = Problem(objective_function)
    agent.set_problem(problem)

where the variable :code:`objective_function` is the agent's objective function in the cost-coupled problem.

Then, the algorithm can be run just like in the Consensus case::

    algorithm = SubgradientMethod(agent=agent, initial_condition=x0)
    algorithm.run(iterations=100)
    print("Output of agent {}: {}".format(agent.id, algorithm.get_result()))

and on the terminal:

.. code-block:: bash

    mpirun -np 8 python script.py

