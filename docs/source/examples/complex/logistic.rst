Cost-coupled: classification via Logistic Regression
===========================================================

For the :ref:`cost-coupled <tutorial>` set-up, we consider a classification scenario [NedOz09]_.
In this example, a linear model is trained by minimizing the so-called *logistic loss functions*.
The complete code of this example is given :ref:`at the end of this page <logistic_regression_code>`.

Problem formulation
--------------------------------------

Suppose there are :math:`N` agents, where each agent :math:`i` is equipped with :math:`m_i`
points :math:`p_{i,1}, \ldots, p_{i,m_i} \in \mathbb{R}^d`
(which represent training samples in a :math:`d`-dimensional feature space). Moreover, suppose
the points are associated to binary labels, that is, each point :math:`p_{i,j}` is labeled with
:math:`\ell_{i,j} \in \{-1,1\}`, for all :math:`j \in \{1, \ldots, m_i\}` and :math:`i \in \{1, \ldots, N\}`.
The problem consists of building a linear classification model from the training
samples by maximizing the a-posteriori probability of each class.
In particular, we look for a separating hyperplane of the form
:math:`\{ z \in \mathbb{R}^d \mid w^\top z + b = 0 \}`, whose parameters
:math:`w` and :math:`b` can be determined by solving the convex optimization problem

.. math::

  \min_{w, b} \: \sum_{i=1}^N \: \sum_{j=1}^{m_i}
  \log \left[ 1 + e^{ -(w^\top p_{i,j} + b) \ell_{i,j} } \right] + \frac{C}{2} \|w\|^2,


where :math:`C > 0` is a parameter affecting regularization. Notice that the problem
is cost coupled (refer to the :ref:`general formulation <tutorial>`),
with each local cost function :math:`f_i` given by

.. math::
  f_i(w, b) = \sum_{j=1}^{m_i} 
  \log \left[ 1 + e^{ -(w^\top p_{i,j} + b) \ell_{i,j} } \right] + \frac{C}{2N} \|w\|^2,
  \hspace{1cm}
  i \in \{1, \ldots, N\}.

The goal is to make agents agree on a common solution :math:`(w^\star, b^\star)`,
so that all of them can compute the separating hyperplane as
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

.. We run a comparative study with :math:`N = 30` agents with the following distributed algorithms:

.. * :ref:`Distributed Subgradient <alg_subgradient>`
.. * :ref:`Gradient Tracking <alg_gradient_tracking>`
.. * :ref:`Distributed Dual Decomposition <alg_dual_decomp>`
.. * :ref:`Distributed ADMM <alg_admm>`

.. The optimization problem is unconstrained, however we add an artificial bounding box
.. :math:`X = \{(w, b) \mid -10 \le w, b \le 10\}` to meet the assumptions of the two latter
.. algorithms, i.e., that the constraint set must be compact.

.. As for the step size, we use the following rules:

.. * constant step-size :math:`\alpha^k = 0.01` - for gradient tracking and distributed ADMM
.. * diminishing step-size :math:`\alpha^k = \frac{1}{k^{0.6}}` - for distributed subgradient and distributed dual decomposition

.. In the following figures we show the evolution of the four algorithms, compared to
.. the solution obtained with a centralized solver. TODO figures:

.. * cost convergence
.. * point convergence
.. * separating hyperplane

Complete code
--------------------------------------
.. _logistic_regression_code:

.. literalinclude:: ../../../../examples/setups/logistic_regression/launcher.py
  :caption: examples/setups/logistic_regression/launcher.py

.. literalinclude:: ../../../../examples/setups/logistic_regression/results.py
  :caption: examples/setups/logistic_regression/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 20 --oversubscribe python launcher.py
  > python results.py

.. rubric:: References

.. [NedOz09] Nedic, Angelia; Asuman Ozdaglar: Distributed subgradient methods for multi-agent optimization: IEEE Transactions on Automatic Control 54.1 (2009): 48.