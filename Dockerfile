FROM florianwechsung/firedrake:latest

# Install

SHELL ["/bin/bash", "-c"]
RUN ls
RUN ls /
RUN source firedrake/bin/activate; which python
RUN cd ~
RUN which python
RUN pip3 install roltrilinos
RUN pip3 install ROL

RUN mkdir -p /src/
WORKDIR /src/
COPY . /src/
RUN ls
RUN pytest
