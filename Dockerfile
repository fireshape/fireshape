FROM ubuntu:17.10

# Install
RUN apt-get -qq update
RUN apt-get install -y curl
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
RUN python3 ./firedrake-install --disable-ssh --minimal-petsc 
