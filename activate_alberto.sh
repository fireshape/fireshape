path_remove /opt/local/bin;
path_remove /opt/local/sbin;
path_remove /Library/Frameworks/Python.framework/Versions/3.5/bin

. ~/Documents/FIREDRAKE/firedrake3/bin/activate
export PYTHONPATH=~/bin/pyrol/src/cmake-install
export PYTHONPATH=$(pwd):$PYTHONPATH
