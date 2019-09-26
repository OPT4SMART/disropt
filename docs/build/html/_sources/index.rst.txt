.. disropt documentation master file, created by
   sphinx-quickstart on Wed Jul  3 13:01:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to disropt
=====================================

**disropt** is a Python package developed within the excellence research program ERC in the project `OPT4SMART <http://www.opt4smart.eu>`_.
The aim of this package is to provide an easy way to run distributed optimization algorithms that can 
be executed by a network of peer copmuting systems.

A comprehensive guide to **disropt** can be found in the :ref:`tutorial`. Many examples are provided in the :ref:`examples` section, while the :ref:`api_documentation` can be checked
for more details.
The package is equipped with some commonly used objective functions and constraints which can be directly used.
 
**disropt** currently supports MPI in order to emulate peer-to-peer communication. However, custom communication protocols can be also implemented.

For example, the following Python code generates an unconstrained, quadratic optimization problem with a 5-dimensional decision variable, 
which is solved using the so-called :ref:`Gradient Tracking <alg_gradient_tracking>`.

.. literalinclude:: ../../examples/basic_example.py
    :caption: basic_example.py
    :name: basic_example-py

This code can be executed over a network with 8 agents by issuing the following command:

.. code-block:: bash

    mpirun -np 8 python example.py

.. toctree::
   :hidden:

   Home <self>
   

.. toctree::
    :hidden:

    install/index.rst


.. toctree::
    :hidden:

    quickstart/index.rst


.. toctree::
    :hidden:

    tutorial/index.rst


.. toctree::
    :hidden:

    examples/index.rst


.. toctree::
    :hidden:

    api_documentation/index.rst

.. toctree::
    :hidden:

    advanced/index.rst

.. toctree::
    :hidden:

    acknowledgements/index.rst