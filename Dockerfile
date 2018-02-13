FROM ubuntu:17.10

# Install
RUN apt-get -qq update
RUN apt-get install -y curl
RUN apt-get install -y python3-dev python3-pip python3-scipy cmake python3-venv python3-tk
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
RUN python3 ./firedrake-install --disable-ssh --minimal-petsc 
