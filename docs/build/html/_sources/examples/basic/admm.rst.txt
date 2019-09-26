Distributed ADMM
=======================

.. literalinclude:: ../../../../examples/algorithms/distributed_ADMM/launcher.py
  :caption: examples/setups/distributed_ADMM/launcher.py

.. literalinclude:: ../../../../examples/algorithms/distributed_ADMM/results.py
  :caption: examples/setups/distributed_ADMM/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py