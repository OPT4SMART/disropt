import setuptools

with open("README_pkg.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="disropt",
    version="0.1.9",
    author="Francesco Farina, Andrea Camisa, Andrea Testa, Ivano Notarnicola, Giuseppe Notarstefano",
    author_email="franc.farina@unibo.it",
    description="DISROPT: a python framework for distributed optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://opt4smart.github.io/disropt/",
    packages=setuptools.find_packages(),
    install_requires=[
        'autograd>=1.3',
        'cvxopt>=1.2.3',
        'cvxpy>=1.0.25',
        'mpi4py>=3.0.1',
        'numpy>=1.16.3',
        'osqp>=0.6.1',
        'scipy>=1.2.1',
        'dill>=0.3.0'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)