Distributed Dual Subgradient
============================

.. warning::
  This example is currently under development

This is an example on how to use the :class:`DualSubgradientMethod` class. See also the reference [FaMa17]_.

.. literalinclude:: ../../../../examples/algorithms/distributed_dual_subgradient/launcher.py
  :caption: examples/setups/distributed_dual_subgradient/launcher.py

.. literalinclude:: ../../../../examples/algorithms/distributed_dual_subgradient/results.py
  :caption: examples/setups/distributed_dual_subgradient/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py