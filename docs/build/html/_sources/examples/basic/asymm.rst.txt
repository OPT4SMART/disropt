ASYMM
=======================

.. literalinclude:: ../../../../examples/algorithms/asymm/launcher.py
  :caption: examples/setups/asymm/launcher.py

.. literalinclude:: ../../../../examples/algorithms/asymm/results.py
  :caption: examples/setups/asymm/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py