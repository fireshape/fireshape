# Fireshape - Shape Optimization with Firedrake

## Documentation
The documentation is available [here](https://fireshape.readthedocs.io/en/latest/index.html#).

## Requirements

Please install the [firedrake finite element library](https://www.firedrakeproject.org) first.
You will need Firedrake together with pyadjoint, i.e.

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install --install pyadjoint


## Installation
Activate the Firedrake virtualenv first

    source path/to/firedrake/bin/activate

Then install the _Rapid Optimization Library_

    pip3 install --no-cache-dir roltrilinos rol

Now you are ready to install fireshape.

For users: 

    pip install git+https://github.com/fireshape/fireshape.git

For developers:
    
    git clone git@github.com:fireshape/fireshape.git
    cd fireshape
    pip install -e .
