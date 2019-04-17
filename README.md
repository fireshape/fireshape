# Fireshape - Shape Optimization with Firedrake

## Documentation
The documentation is available [here](https://fireshape.readthedocs.io/en/latest/index.html#).

## Requirements

Please install the [firedrake finite element library](https://www.firedrakeproject.org) first.
You will need firedrake together with pyadjoint, i.e.

    python3 firedrake-install --install pyadjoint


## Installation
Activate the firedrake virtualenv first and then run either:

For users: 

    pip install git+https://github.com/fireshape/fireshape.git

For developers:
    
    git clone git@github.com:fireshape/fireshape.git
    cd fireshape
    pip install -e .
