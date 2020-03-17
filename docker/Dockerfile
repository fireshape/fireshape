# DockerFile for a firedrake (modified from the dockerfile provided by the firedrake project)

FROM ubuntu:18.04

# This DockerFile is looked after by
MAINTAINER Florian Wechsung <wechsung@nyu.edu>

# Update and install required packages for Firedrake
USER root
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata \
    && apt-get -y install curl vim \
                 openssh-client build-essential autoconf automake \
                 cmake gfortran git libblas-dev liblapack-dev \
                 libmpich-dev libtool mercurial mpich\
                 python3-dev python3-pip python3-tk python3-venv \
                 python3-requests zlib1g-dev libboost-dev sudo bison flex \
                 gmsh patchelf docker.io \
    && rm -rf /var/lib/apt/lists/*


# Set up user so that we do not run as root
RUN useradd -m -s /bin/bash -G sudo firedrake && \
    echo "firedrake:docker" | chpasswd && \
    echo "firedrake ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    ldconfig

USER firedrake
WORKDIR /home/firedrake

# Now install firedrake
RUN echo "2020-03-17"
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
RUN bash -c "python3 firedrake-install --no-package-manager --disable-ssh --mpicc mpicc.mpich --mpicxx mpicxx.mpich --mpif90 mpif90.mpich --mpiexec mpiexec.mpich"
RUN . /home/firedrake/firedrake/bin/activate; pip3 install wheel --upgrade
RUN . /home/firedrake/firedrake/bin/activate; pip3 install scipy
RUN . /home/firedrake/firedrake/bin/activate; pip3 install roltrilinos
RUN . /home/firedrake/firedrake/bin/activate; pip3 install ROL
