FROM florianwechsung/firedrake:latest

# Install
run ls
run pip3 install roltrilinos
run pip3 install ROL
run pytest
