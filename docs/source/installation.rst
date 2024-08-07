.. _installation:

Installation
============

Requirements
^^^^^^^^^^^^

Please install the finite element library `Firedrake <https://www.firedrakeproject.org/download.html>`_ first.

.. code-block:: bash

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install

How to install Fireshape
^^^^^^^^^^^^^^^^^^^^^^^^

Activate Firedrake's virtualenv first.

.. code-block:: bash

    source path/to/firedrake/bin/activate

Then install the `Rapid Optimization Library <https://trilinos.org/packages/rol/>`_ along with :code:`roltrilinos`.

.. code-block:: bash

    pip3 install --no-cache-dir pyroltrilinos

Now you are ready to install fireshape.

For users:

.. code-block:: bash

    pip3 install git+https://github.com/fireshape/fireshape.git

For developers:

.. code-block:: bash

    git clone git@github.com:fireshape/fireshape.git
    pip install -e fireshape/
