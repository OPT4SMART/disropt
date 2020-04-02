Constraints Consensus
==============================

This is an example on how to use the :class:`ConstraintsConsensus` class. See also the reference [NoBu11]_.

.. literalinclude:: ../../../../examples/algorithms/constraints_consensus/launcher.py
  :caption: examples/setups/constraints_consensus/launcher.py

.. literalinclude:: ../../../../examples/algorithms/constraints_consensus/results.py
  :caption: examples/setups/constraints_consensus/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py