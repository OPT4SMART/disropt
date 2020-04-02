Distributed Dual Decomposition
==============================

This is an example on how to use the :class:`DualDecomposition` class.

.. literalinclude:: ../../../../examples/algorithms/distributed_dual_decomposition/launcher.py
  :caption: examples/setups/distributed_dual_decomposition/launcher.py

.. literalinclude:: ../../../../examples/algorithms/distributed_dual_decomposition/results.py
  :caption: examples/setups/distributed_dual_decomposition/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py