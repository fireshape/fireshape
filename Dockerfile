FROM firedrakeproject/firedrake:latest

# Install

RUN . firedrake/bin/activate; pip3 install wheel --upgrade
RUN . firedrake/bin/activate; pip3 install scipy
RUN . firedrake/bin/activate; pip3 install roltrilinos
RUN . firedrake/bin/activate; pip3 install ROL

RUN mkdir -p /home/firedrake/src/
WORKDIR /home/firedrake/src/
COPY . /home/firedrake/src/
RUN . /home/firedrake/firedrake/bin/activate; export PYTHONPATH=$(pwd):$PYTHONPATH; pytest
