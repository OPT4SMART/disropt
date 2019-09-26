.. _tutorial_algorithms:

Algorithms
===================================

In **disropt**, there are many implemented distributed optimization algorithms. Each algorithm is tailored
for a specific distributed optimization set-up (see :ref:`tutorial`).

Basic
^^^^^^^^^^^
* :ref:`Consensus <alg_consensus>` (standard and block wise, synchronous and asynchronous)

Cost-coupled set-up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :ref:`Distributed Subgradient <alg_subgradient>` (standard and block wise)
* :ref:`Gradient Tracking <alg_gradient_tracking>`
* :ref:`Distributed Dual Decomposition <alg_dual_decomp>`
* :ref:`Distributed ADMM <alg_admm>`
* :ref:`ASYMM <alg_asymm>`

Common cost set-up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :ref:`Constraints Consensus <alg_constraintsconsensus>`
* :ref:`Set membership <alg_set_membership>` (synchronous and asynchronous)

Constraint-coupled set-up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :ref:`Distributed Dual Subgradient <alg_dual_subgradient>`
* :ref:`Distributed Primal Decomposition <alg_primal_decomp>`

Miscellaneous
^^^^^^^^^^^^^^^^^
* :ref:`Logic AND <alg_logic_and>` (synchronous and asynchronous)
