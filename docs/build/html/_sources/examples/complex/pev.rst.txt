Constraint-coupled: charging of Plug-in Electric Vehicles (PEVs)
================================================================

For the :ref:`constraint-coupled <tutorial>` set-up, we consider the problem
of determining an optimal overnight charging schedule for a fleet of Plug-in Electric Vehicles (PEVs).
The model described in this page is inspired by the model in the paper [VuEs16]_ (we consider the
"charge-only" case without the integer constraints on the input variables).
The complete code of this example is given :ref:`at the end of this page <pev_code>`.

Problem formulation
--------------------------------------

Suppose there is a fleet of :math:`N` PEVs (agents) that must be charged by drawing power from
the same electricity distribution network. Assuming the vehicles are connected to the grid
at a certain time (e.g., at midnight), the goal is to determine an optimal overnight schedule
to charge the vehicles, since the electricity price varies during the charging period.

Formally, we divide the entire charging period into a total of :math:`T = 24` time slots,
each one of duration :math:`\Delta T = 20` minutes. For each PEV :math:`i \in \{1, \ldots, N\}`,
the charging power at time step :math:`k` is equal to :math:`P_i u_i(k)`, where :math:`u_i(k) \in [0, 1]`
is the input to the system and :math:`P_i` is the maximum charging power that can be fed to the
:math:`i`-th vehicle.

System model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The state of charge of the :math:`i`-th battery is denoted by :math:`e_i(k)`,
its initial state of charge is :math:`E_i^\text{init}`, which by the end of the charging
period has to attain at least :math:`E_i^\text{ref}`. The charging conversion efficiency
is denoted by :math:`\zeta_i^\text{u} \triangleq 1 - \zeta_i`, where :math:`\zeta_i > 0`
encodes the conversion losses. The battery's capacity limits are denoted by
:math:`E_i^\text{min}, E_i^\text{max} \ge 0`. The system's dynamics are therefore given by

.. math::

  & \: e_i(0) = E_i^\text{init}
  \\
  & \: e_i(k+1) = e_i(k) + P_i \Delta T \zeta_i^u u_i(k), \hspace{0.5cm} k \in \{0, \ldots, T-1\}
  \\
  & \: e_i(T) \ge E_i^\text{ref}
  \\
  & \: E_i^\text{min} \le e_i(k) \le E_i^\text{max}, \hspace{2.88cm} k \in \{1, \ldots, T\}
  \\
  & \: u_i(k) \in [0,1], \hspace{4.47cm} k \in \{0, \ldots, T-1\}.

To model congestion avoidance of the power grid, we further consider the following (linear)
coupling constraints among all the variables

.. math::

  \sum_{i=1}^N P_i u_i(k) \le P^\text{max}, \hspace{1cm} k \in \{0, \ldots, T-1\},

where :math:`P^\text{max}` is the maximum power that the be drawn from the grid.

Optimization problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We assume that, at each time slot :math:`k`, electricity has unit cost equal to
:math:`C^\text{u}(k)`. Since the goal is to minimize the overall consumed energy price,
the global optimization problem can be posed as

.. math::

  \min_{u, e} \: & \: \sum_{i=1}^N \sum_{k=0}^{T-1} C^\text{u}(k) P_i u_i(k)
  \\
  \text{subject to} \: & \: \sum_{i=1}^N P_i u_i(k) \le P^\text{max}, \hspace{1cm} k \in \{0, \ldots, T-1\}
  \\
  & \: (u_i, e_i) \in X_i, \hspace{2.4cm} \: i \in \{1, \ldots, N\}.

The problem is recognized to be a :ref:`constraint-coupled <tutorial>` problem,
with local variables :math:`x_i` equal to the stack of :math:`e_i(k), u_i(k)` for :math:`k \in \{0, \ldots, T-1\}`,
plus :math:`e_i(T)`. The local objective function is equal to

.. math::

  f_i(x_i) = \sum_{k=0}^{T-1} P_i u_i(k) C^\text{u}(k),

the local constraint set is equal to

.. math::

  X_i = \{(e_i, u_i) \in \mathbb{R}^{T+1} \times \mathbb{R}^T \text{ such that local dynamics is satisfied} \}

and the local coupling constraint function :math:`g_i : \mathbb{R}^{2T+1} \rightarrow \mathbb{R}^T` has components

.. math::

  g_{i,k}(x_i) = P_i u_i(k) - \frac{P^\text{max}}{N}, \hspace{1cm} k \in \{0, \ldots, T-1\}.

The goal is to make each agent compute its portion :math:`x_i^\star = (e_i^\star, u_i^\star)`
of an optimal solution :math:`(x_1^\star, \ldots, x_N^\star)` of the optimization problem,
so that all of them can know their own assignment of the optimal charging schedule, given by
:math:`(u_i^\star(0), \ldots, u_i^\star(T-1))`.


Data generation model
--------------------------------------

The data are generated according to table in [VuEs16]_ (see Appendix).

.. Simulation results
.. --------------------------------------

.. We run a comparative study with :math:`N = 50` agents with the following distributed algorithms:

.. * :ref:`Distributed Dual Subgradient <alg_dual_subgradient>`
.. * :ref:`Distributed Primal Decomposition <alg_primal_decomp>`

.. For the Distributed Primal Decomposition algorithm, we choose a sufficiently large parameter :math:`M = 100`.
.. As for the step-size, we use for both algorithms the diminishing rule :math:`\alpha^k = \frac{1}{k^{0.6}}`.

.. In the following figures we show the evolution of the two algorithms...... TODO figures:

.. * cost convergence
.. * coupling constraint value

Complete code
--------------------------------------
.. _pev_code:

.. literalinclude:: ../../../../examples/setups/pev/launcher.py
  :caption: examples/setups/pev/launcher.py

.. literalinclude:: ../../../../examples/setups/pev/results.py
  :caption: examples/setups/pev/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 50 --oversubscribe python launcher.py
  > python results.py

.. rubric:: References
.. [VuEs16] Vujanic, R., Esfahani, P. M., Goulart, P. J., Mariéthoz, S., & Morari, M. (2016). A decomposition method for large scale MILPs, with performance guarantees and a power system application. Automatica, 67, 144-156.