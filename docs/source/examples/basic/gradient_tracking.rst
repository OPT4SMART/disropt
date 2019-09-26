Gradient Tracking
=======================

.. literalinclude:: ../../../../examples/algorithms/gradient_tracking/launcher.py
  :caption: examples/setups/gradient_tracking/launcher.py

.. literalinclude:: ../../../../examples/algorithms/gradient_tracking/results.py
  :caption: examples/setups/gradient_tracking/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py