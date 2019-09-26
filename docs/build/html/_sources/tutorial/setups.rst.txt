.. _tutorial_setups:

Distributed optimization set-ups
===================================
Distributed optimization problems usually arising in applications usually enjoy a proper structure in their mathematical formulation.
In **disropt**, three different optimization set-ups are available. As for their solution, see the :ref:`Algorithms <tutorial_algorithms>` page 
for a list of all the implemented distributed algorithms (classified by optimization set-up to which they apply).

Cost-coupled set-up
^^^^^^^^^^^^^^^^^^^^^^^^^
In this optimization set-up, the cost function is expressed as the sum of cost functions :math:`f_i`
and all of them depend on a common optimization variable :math:`x`. Formally, the set-up is

.. math::
    \min_{x} \: & \: \sum_{i=1}^N f_i(x)
    \\
    \text{subject to} \: & \: x \in X,

where :math:`x \in \mathbb{R}^d` and :math:`X \subseteq \mathbb{R}^d`. 
The global constraint set :math:`X` is common to all agents, while :math:`f_i : \mathbb{R}^d \rightarrow \mathbb{R}`
is assumed to be known by agent :math:`i` only, for all :math:`i\in \{1, \ldots, N\}`.

In some applications, the constraint set :math:`X` can be expressed as the intersection of local constraint sets, i.e.,

.. math::
    X = \bigcap_{i=1}^N X_i

where each :math:`X_i \subseteq \mathbb{R}^d` is meant to be known by agent :math:`i` only,
for all :math:`i\in \{1, \ldots, N\}`.

The goal for distributed algorithms for the cost-coupled set-up is that all agent estimates
are eventually consensual to an optimal solution :math:`x^\star` of the problem.

Common cost set-up
^^^^^^^^^^^^^^^^^^^^^^^^^
In this optimization set-up, there is a unique cost function :math:`f` that depends on a common
optimization variable :math:`x`, and the optimization variable must further satisfy local constraints.
Formally, the set-up is

.. math::
    \min_{x} \: & \: f(x)
    \\
    \text{subject to} \: & \: x \in \bigcap_{i=1}^N X_i,

where :math:`x \in \mathbb{R}^d` and each :math:`X_i \subseteq \mathbb{R}^d`. The cost function :math:`f`
is assumed to be known by all the agents, while each set :math:`X_i` is assumed to be known by agent
:math:`i` only, for all :math:`i\in \{1, \ldots, N\}`.

.. The goal for distributed algorithms for the common-cost set-up is to asymptotically reach an agreement among all the agents on an optimal solution :math:`x^\star` of the optimization problem. ...

The goal for distributed algorithms for the common-cost set-up is that all agent estimates
are eventually consensual to an optimal solution :math:`x^\star` of the problem.


Constraint-coupled set-up
^^^^^^^^^^^^^^^^^^^^^^^^^
In this optimization set-up, the cost function is expressed as the sum of local cost functions :math:`f_i`
that depend on a local optimization variable :math:`x_i`. The variables must satisfy local constraints (involving only each 
optimization variable :math:`x_i`) and global coupling constraints (involving all the optimization variables).
Formally, the set-up is

.. math::
    \min_{x_1,\ldots,x_N} \: & \: \sum_{i=1}^N f_i(x_i)
    \\
    \text{subject to} \: & \: x_i \in X_i, \hspace{1cm}  i \in \{1, \ldots, N\}
    \\
    & \: \sum_{i=1}^N g_i (x_i) \le 0,

where each :math:`x_i \in \mathbb{R}^{d_i}`, :math:`X_i \subseteq \mathbb{R}^{d_i}`,
:math:`f_i : \mathbb{R}^{d_i} \rightarrow \mathbb{R}` and :math:`g_i : \mathbb{R}^{d_i} \rightarrow \mathbb{R}^S`
for all :math:`i \in \{1, \ldots, N\}`. Here the symbol :math:`\le` is also used to denote component-wise
inequality for vectors. Therefore, the optimization variable consists of the stack of all :math:`x_i`, namely the vector :math:`(x_1,\ldots,x_N)`.
All the quantities with the index :math:`i` are assumed
to be known by agent :math:`i` only, for all :math:`i\in \{1, \ldots, N\}`. The function :math:`g_i`,
with values in :math:`\mathbb{R}^S`, is used to express the :math:`i`-th contribution to
:math:`S` coupling constraints among all the variables.

The goal for distributed algorithms for the constraint-coupled set-up is that each agent estimate
is asymptotically equal to its portion :math:`x_i^\star \in X_i` of an optimal solution :math:`(x_1^\star, \ldots, x_N^\star)`
of the problem.