FROM florianwechsung/firedrake:latest

# Install

RUN . firedrake/bin/activate; pip3 install roltrilinos
RUN . firedrake/bin/activate; pip3 install ROL
RUN mkdir -p /src/
WORKDIR /src/
COPY . /src/
RUN . firedrake/bin/activate; pytest
