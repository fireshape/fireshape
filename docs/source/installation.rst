.. _installation:

Installation
============

Requirements
^^^^^^^^^^^^

Please install the `firedrake finite element library <https://www.firedrakeproject.org/download.html>`_ first.

How to install Fireshape
^^^^^^^^^^^^^^^^^^^^^^^^

Activate the firedrake virtualenv first and then run either:

For users:

.. code-block:: bash

    pip install git+https://github.com/fireshape/fireshape.git

For developers:

.. code-block:: bash

    git clone git@github.com:fireshape/fireshape.git
    cd fireshape
    pip install -e .
