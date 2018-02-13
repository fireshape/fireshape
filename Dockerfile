FROM florianwechsung/firedrake:latest

# Install
run . firedrake/bin/activate
run pip3 install roltrilinos
run pip3 install ROL
