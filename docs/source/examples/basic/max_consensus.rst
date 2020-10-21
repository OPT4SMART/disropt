Distributed Max Consensus
===========================

This is an example on how to use the :class:`MaxConsensus` class.

.. literalinclude:: ../../../../examples/algorithms/max-consensus/launcher.py
  :caption: examples/setups/max-consensus/launcher.py

The file can be executed by issuing the following command in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py