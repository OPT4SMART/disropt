<!-- <div align="center">
<img src="./docs/source/_static/blocks_bigdata_t.png" alt="logo" width="250px"></img>
</div> -->

# Welcome to DISROPT
[**Reference docs**](...)
| [**Install guide**](#installation)
| [**Quickstart**](#quickstart)

**DISROPT** is a Python package developed within the excellence research program ERC in the project [OPT4SMART](www.opt4smart.eu).
The aim of this package is to provide an easy way to run distributed optimization algorithms that can 
be executed by a network of peer copmuting systems.

A comprehensive guide to DISROPT can be found in the [documentation](...)
The package is equipped with some commonly used objective functions and constraints which can be directly used.
 
**DISROPT** currently supports MPI in order to emulate peer-to-peer communication. However, custom communication protocols can be also implemented.

## Installation
**DISROPT** works on Python 3.7 and requires an MPI implementation to be installed on your system.

It relies on

* numpy
* scipy
* mpi4py
* autograd
* osqp (optional, but required for some features)
* cvxopt (optional, but required for some features)
* cvxpy (optional, but required for some features)

### pip installation
```
pip install disropt
```

## Quickstart
* [Distributed optimization introduction](...)
* [Using functions and constraints](...)
* [Defining problems](...)
* [Creating agents](...)
* [Running algorithmss](...)

## Citing **DISROPT**
TODO...

## Constributors
**DISROPT** is developed by
[Francesco Farina](https://francescofarina.github.io),
[Andrea Camisa](https://www.unibo.it/sitoweb/a.camisa),
[Andrea Testa](http://opt4smart.dei.unibo.it/people/andrea_testa.html),
[Ivano Notarnicola](https://www.unibo.it/sitoweb/ivano.notarnicola), and
[Giuseppe Notarstefano](https://www.unibo.it/sitoweb/giuseppe.notarstefano)

## Acknoledgements
This result is part of a project that has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 638992 - OPT4SMART).

<p style="text-align:center">
  <img src="./docs/source/_static/logo_ERC.png" width="200" />
  <img src="./docs/source/_static/logo_OPT4Smart.png" width="200" /> 
</p>