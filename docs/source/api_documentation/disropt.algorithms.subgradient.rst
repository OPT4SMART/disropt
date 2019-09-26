(Sub)gradient based Algorithms
==============================

Distributed projected (Sub)gradient Method
-------------------------------------------
.. _alg_subgradient:

.. autoclass:: disropt.algorithms.subgradient.SubgradientMethod
   :members:
   :undoc-members:
   :show-inheritance:


Randomized Block (Sub)gradient Method
--------------------------------------

.. autoclass:: disropt.algorithms.subgradient.BlockSubgradientMethod
   :members:
   :undoc-members:
   :show-inheritance:
   

Distributed Gradient Tracking
--------------------------------------
.. _alg_gradient_tracking:

.. autoclass:: disropt.algorithms.gradient_tracking.GradientTracking
   :members:
   :undoc-members:
   :show-inheritance:


Distributed Gradient Tracking (over directed, unbalanced graphs)
-----------------------------------------------------------------

.. autoclass:: disropt.algorithms.gradient_tracking.DirectedGradientTracking
   :members:
   :undoc-members:
   :show-inheritance:


.. rubric:: References

.. [NeOz09] Nedic, Angelia; Asuman Ozdaglar: Distributed subgradient methods for multi-agent optimization: IEEE Transactions on Automatic Control 54.1 (2009): 48.
.. [FaNo19] Farina, Francesco, and Notarstefano, Giuseppe. Randomized Block Proximal Methods for Distributed Stochastic Big-Data Optimization. arXiv preprint arXiv:1905.04214 (2019).
.. [XiKh18] Xin, Ran, and Usman A. Khan: A linear algorithm for optimization over directed graphs with geometric convergence. IEEE Control Systems Letters 2.3 (2018): 315-320.