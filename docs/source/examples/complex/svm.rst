Common-cost: classification via Support Vector Machine
===========================================================

For the :ref:`common-cost <tutorial>` set-up, we consider a classification scenario.
In this example, a linear model is trained by maximizing the distance of the separating
hyperplane from the training points.
The complete code of this example is given :ref:`at the end of this page <svm_code>`.

.. warning::
  This example is currently under development

Problem formulation
--------------------------------------

Suppose there are :math:`N` agents, where each agent :math:`i` is equipped with :math:`m_i`
points :math:`p_{i,1}, \ldots, p_{i,m_i} \in \mathbb{R}^d`
(which represent training samples in a :math:`d`-dimensional feature space). Moreover, suppose
the points are associated to binary labels, that is, each point :math:`p_{i,j}` is labeled with
:math:`\ell_{i,j} \in \{-1,1\}`, for all :math:`j \in \{1, \ldots, m_i\}` and :math:`i \in \{1, \ldots, N\}`.
The problem consists of building a classification model from the training samples.
In particular, we look for a separating hyperplane of the form
:math:`\{ z \in \mathbb{R}^d \mid w^\top z + b = 0 \}` such that it separates all the points
with :math:`\ell_i = -1` from all the points with :math:`\ell_i = 1`.

In order to maximize the distance of the separating hyperplane from the training points,
one can solve the following (convex) quadratic program:

.. math::

  \min_{w, b, \xi} \: & \: \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \sum_{j=1}^{m_i} \xi_{i,j}
  \\
  \text{subject to} \: & \: \ell_{i,j} ( w^\top p_{i,j} + b ) \ge 1 - \xi_{i,j}, \hspace{1cm} \forall \: j, i
  \\
  & \: \xi \ge 0,

where :math:`C > 0` is a parameter affecting regularization. The optimization problem
above is called *soft-margin SVM* since it allows for the presence of outliers by
activating the variables :math:`\xi_{i,j}` in case a separating hyperplane does not exist.
Notice that the problem can be viewed either as a common-cost problem or as a cost-coupled
problem(refer to the :ref:`general formulations <tutorial>`).
Here we consider the problem as a common cost, with the common objective function equal to

.. math::
  f(w, b, \xi) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \sum_{j=1}^{m_i} \xi_{i,j}

and each local constraint set :math:`X_i` given by

.. math::
  X_i = \{ (w, b, \xi) \mid \xi \ge 0, \ell_{i,j} ( w^\top p_{i,j} + b ) \ge 1 - \xi_{i,j}, \text{ for all } j \in \{1, \ldots, m_i\} \}.

The goal is to make agents agree on a common solution :math:`(w^\star, b^\star, \xi^\star)`,
so that all of them can compute the soft-margin separating hyperplane as
:math:`\{ z \in \mathbb{R}^d \mid (w^\star)^\top z + b^\star = 0 \}`.

Data generation model
--------------------------------------

We consider a bidimensional sample space (:math:`d = 2`).
Agents generate a certain number of samples (between 2 and 5) for both labels.
For each label, the samples are drawn according to a multivariate gaussian distribution,
with covariance matrix equal to the identity and mean equal to :math:`(0,0)`
(for the label :math:`1`) and :math:`(3,2)` (for the label :math:`-1`).
The regularization parameter is set to :math:`C = 10`.

.. Simulation results
.. --------------------------------------

.. We run a simulation study with :math:`N = 30` agents with the :ref:`Constraints Consensus <alg_constraintsconsensus>`
.. distributed algorithm.

.. TODO TODO TODO TODO TODO TODO TODO

Complete code
--------------------------------------
.. _svm_code:

.. literalinclude:: ../../../../examples/setups/svm/launcher.py
  :caption: examples/setups/svm/launcher.py

.. literalinclude:: ../../../../examples/setups/svm/results.py
  :caption: examples/setups/svm/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py
