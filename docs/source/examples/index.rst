.. _examples:

Examples
***********************

Here we report some examples that show how to use **disropt**.
We divide the examples into two groups:

- :ref:`Basic examples <examples_basic>`: to show how each algorithm should be used
- :ref:`Complex examples <examples_complex>`: realistic application scenarios, e.g., to make comparative studies on different algorithms

Basic examples
#######################
.. _examples_basic:

We provide an example for each implemented distributed algorithm
(see also the :ref:`list of implemented algorithms <tutorial_algorithms>`).

.. toctree::
    :maxdepth: 1

    Consensus <basic/consensus.rst>
    Logic AND <basic/logic_and.rst>
    Set Membership <basic/set_membership.rst>
    Distributed Subgradient <basic/subgradient.rst>
    Gradient Tracking <basic/gradient_tracking.rst>
    Distributed Dual Decomposition <basic/dual_decomposition.rst>
    Distributed ADMM <basic/admm.rst>
    ASYMM <basic/asymm.rst>
    Constraints Consensus <basic/constraints_consensus.rst>
    Distributed Dual Subgradient <basic/dual_subgradient.rst>
    Distributed Primal Decomposition <basic/primal_decomposition.rst>

Complex examples
#######################
.. _examples_complex:

We provide three complex examples on realistic applications, one for each
optimization set-up (see also the :ref:`tutorial introduction <tutorial>`).

.. toctree::
    :maxdepth: 1

    Cost coupled: classification via Logistic Regression <complex/logistic.rst>
    Common cost: classification via Support Vector Machine <complex/svm.rst>
    Constraint coupled: charging of Plug-in Electric Vehicles <complex/pev.rst>