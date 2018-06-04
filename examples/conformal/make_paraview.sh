#!/bin/bash
pvpython='/Applications/ParaView-5.4.1.app/Contents/bin/pvpython'
$pvpython plot_paraview.py --base_inner elasticity --use_cr 1 
$pvpython plot_paraview.py --base_inner elasticity --use_cr 0 
$pvpython plot_paraview.py --base_inner laplace    --use_cr 1 
$pvpython plot_paraview.py --base_inner laplace    --use_cr 0 
