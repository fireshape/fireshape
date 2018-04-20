FROM ubuntu:17.10
MAINTAINER Florian Wechsung <wechsung@maths.ox.ac.uk>

USER root

RUN apt-get -qq update
RUN apt-get install -y curl python3-dev python3-pip python3-scipy cmake python3-venv python3-tk autoconf automake gfortran git-core libblas-dev liblapack-dev libopenmpi-dev libtool mercurial openmpi-bin zlib1g-dev libboost-all-dev patchelf

RUN echo "firedrake as of 2018-04-20"
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
RUN python3 ./firedrake-install --disable-ssh
