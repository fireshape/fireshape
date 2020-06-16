FROM firedrakeproject/firedrake-vanilla:latest

MAINTAINER Alberto Paganini <admp1@le.ac.uk>

USER root
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && apt-get -y install gmsh patchelf \
    && rm -rf /var/lib/apt/lists/*

USER firedrake
RUN . /home/firedrake/firedrake/bin/activate; pip3 install wheel --upgrade
RUN . /home/firedrake/firedrake/bin/activate; pip3 install scipy
RUN . /home/firedrake/firedrake/bin/activate; pip3 install roltrilinos
RUN . /home/firedrake/firedrake/bin/activate; pip3 install ROL
