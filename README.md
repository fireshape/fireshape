# Fireshape - Shape Optimization with Firedrake


[![CircleCI](https://circleci.com/gh/fireshape/fireshape.svg?style=shield)](https://circleci.com/gh/fireshape/fireshape)
[![Read the Docs](https://readthedocs.org/projects/fireshape/badge/?version=latest)](https://fireshape.readthedocs.io/en/latest/)
[![Actions Docker images](https://github.com/fireshape/fireshape/actions/workflows/build.yml/badge.svg)](https://github.com/fireshape/fireshape/actions/workflows/build.yml)

## Documentation
The documentation is available [here](https://fireshape.readthedocs.io/en/latest/index.html#).

We have also published an Open Access paper about Fireshape. It is available
[here](https://doi.org/10.1007/s00158-020-02813-y).

## Requirements

Please install the [firedrake finite element library](https://www.firedrakeproject.org) first by following the instructions [here](https://firedrakeproject.org/install).

## Installation

For users:

    pip3 install git+https://github.com/fireshape/fireshape.git

For developers:

    git clone git@github.com:fireshape/fireshape.git
    python -m pip install -e fireshape/

If you have installed Firedrake into a virtual environment then this must be activated first.

## Docker images
Fireshape is also available as a [docker image](https://hub.docker.com/r/fireshape/fireshape).
