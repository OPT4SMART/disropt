Distributed Simplex
======================================

This is an example on how to use the :class:`DistributedSimplex` class. See also the reference [BuNo11]_.

.. literalinclude:: ../../../../examples/algorithms/distributed_simplex/launcher.py
  :caption: examples/setups/distributed_simplex/launcher.py

.. literalinclude:: ../../../../examples/algorithms/distributed_simplex/results.py
  :caption: examples/setups/distributed_simplex/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 10 --oversubscribe python launcher.py
  > python results.py

Distributed Simplex (dual problem)
======================================

This is an example on how to use the :class:`DualDistributedSimplex` class, which forms the dual
problem of the given optimization problem and solves it with the Distributed Simplex algorithm.

.. literalinclude:: ../../../../examples/algorithms/distributed_simplex/launcher_dual.py
  :caption: examples/setups/distributed_simplex/launcher_dual.py

.. literalinclude:: ../../../../examples/algorithms/distributed_simplex/results_dual.py
  :caption: examples/setups/distributed_simplex/results_dual.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 10 --oversubscribe python launcher_dual.py
  > python results_dual.py