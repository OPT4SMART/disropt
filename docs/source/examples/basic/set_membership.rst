Distributed Set Membership
==========================

This is an example on how to use the :class:`SetMembership` class. See also the reference [FaGa18]_.

.. literalinclude:: ../../../../examples/algorithms/set_membership/launcher.py
  :caption: examples/setups/set_membership/launcher.py

.. literalinclude:: ../../../../examples/algorithms/set_membership/results.py
  :caption: examples/setups/set_membership/results.py

The two files can be executed by issuing the following commands in the example folder:

.. code-block:: bash

  > mpirun -np 30 --oversubscribe python launcher.py
  > python results.py