# Fireshape - Shape Optimization with Firedrake

## Documentation
The documentation is available [here](https://fireshape.readthedocs.io/en/latest/index.html#).

We have recently written a manuscript about Fireshape. It is available [here](https://arxiv.org/abs/2005.07264).

## Requirements

Please install the [firedrake finite element library](https://www.firedrakeproject.org) first.

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install


## Installation
Activate the Firedrake virtualenv first

    source path/to/firedrake/bin/activate

If you're on linux, you will need the patchelf package. Install it e.g. via 

    sudo apt install patchelf

On MacOS, this not necessary. Now install the _Rapid Optimization Library_

    pip3 install --no-cache-dir roltrilinos rol

Now you are ready to install fireshape.

For users: 

    pip install git+https://github.com/fireshape/fireshape.git

For developers:
    
    git clone git@github.com:fireshape/fireshape.git
    cd fireshape
    pip install -e .
