
from paraview.simple import *
import os
import numpy as np


base = r'/Users/florianwechsung/Documents/Uni/DPhil/shapelib/examples/conformal/output/'
hist_maximum = 500

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_cr", type=int, default=1)
parser.add_argument("--base_inner", type=str, default="elasticity")
args = parser.parse_args()

use_cr = bool(args.use_cr)
base_inner = args.base_inner

label = "base_%s_cr_%s" % (base_inner, use_cr)

img_directory = base + "img/"
if not os.path.exists(img_directory):
    os.makedirs(img_directory)

dilatation_pvd = PVDReader(FileName=base + label + "/u.pvd")
animationScene = GetAnimationScene()

animationScene.UpdateAnimationUsingDataTimeSteps()
renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.OrientationAxesVisibility = 0
renderView1.Background = [1, 1, 1]
display = Show(dilatation_pvd, renderView1)
display.Representation = 'Surface'
display.ColorArrayName = [None, '']
# display.OSPRayScaleArray = 'function_16'
# display.OSPRayScaleFunction = 'PiecewiseFunction'
display.SelectOrientationVectors = 'None'
display.SelectScaleArray = 'None'
display.GlyphType = 'Arrow'
# display.PolarAxes = 'PolarAxesRepresentation'
display.ScalarOpacityUnitDistance = 0.417380156334143
# display.GaussianRadius = 0.5
# display.SetScaleArray = ['POINTS', 'function_16']
# display.ScaleTransferFunction = 'PiecewiseFunction'
# display.OpacityArray = ['POINTS', 'function_16']
# display.OpacityTransferFunction = 'PiecewiseFunction'

# change representation type
display.SetRepresentationType('Surface With Edges')
display.LineWidth = 2.0

animationScene.GoToLast()
renderView1.ViewSize = [1200, 800]

renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 1]
renderView1.CameraFocalPoint = [0.0, 0.0, 0]
renderView1.CameraParallelScale = 2.05
SaveScreenshot(img_directory + label + "_mesh.png", view=renderView1, magnification=2, quality=95)
renderView1.ViewSize = [800, 800]
renderView1.CameraPosition = [-0.918, 0.0, 1]
renderView1.CameraFocalPoint = [-0.918, 0.0, 0]
renderView1.CameraParallelScale = 0.3
SaveScreenshot(img_directory + label + "_mesh_zoom.png", view=renderView1, magnification=2, quality=95)
# import sys; sys.exit(1)
renderView1.ResetCamera()

# #changing interaction mode based on data extents
# renderView1.InteractionMode = '2D'
# renderView1.CameraPosition = [0.0, 0.0, 10000.0]

# # update the view to ensure updated data information
renderView1.Update()

# # change representation type
# dilatation_pvd.SetRepresentationType('Surface With Edges')

# # Hide orientation axes
# renderView1.OrientationAxesVisibility = 0

# create a new 'Mesh Quality'
meshQuality1 = MeshQuality(Input=dilatation_pvd)

# get color transfer function/color map for 'Quality'
qualityLUT = GetColorTransferFunction('Quality')

# get opacity transfer function/opacity map for 'Quality'
qualityPWF = GetOpacityTransferFunction('Quality')

# show data in view
meshQuality1Display = Show(meshQuality1, renderView1)
# trace defaults for the display properties.
meshQuality1Display.Representation = 'Surface'
meshQuality1Display.AmbientColor = [0.0, 0.0, 0.0]
meshQuality1Display.ColorArrayName = ['CELLS', 'Quality']
meshQuality1Display.LookupTable = qualityLUT
# meshQuality1Display.OSPRayScaleArray = 'Quality'
# meshQuality1Display.OSPRayScaleFunction = 'PiecewiseFunction'
meshQuality1Display.SelectOrientationVectors = 'None'
meshQuality1Display.ScaleFactor = 0.6000000000000001
meshQuality1Display.SelectScaleArray = 'Quality'
meshQuality1Display.GlyphType = 'Arrow'
# meshQuality1Display.GlyphTableIndexArray = 'Quality'
# meshQuality1Display.DataAxesGrid = 'GridAxesRepresentation'
# meshQuality1Display.PolarAxes = 'PolarAxesRepresentation'
meshQuality1Display.ScalarOpacityFunction = qualityPWF
meshQuality1Display.ScalarOpacityUnitDistance = 0.30725553282259715
# meshQuality1Display.GaussianRadius = 0.30000000000000004
# meshQuality1Display.SetScaleArray = ['POINTS', 'function_10']
# meshQuality1Display.ScaleTransferFunction = 'PiecewiseFunction'
# meshQuality1Display.OpacityArray = ['POINTS', 'function_10']
# meshQuality1Display.OpacityTransferFunction = 'PiecewiseFunction'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
# meshQuality1Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
# meshQuality1Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
# meshQuality1Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
# meshQuality1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
# meshQuality1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
# meshQuality1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
# meshQuality1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
# meshQuality1Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
# meshQuality1Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
# meshQuality1Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
# meshQuality1Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(dilatation_pvd, renderView1)

# show color bar/color legend
meshQuality1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# CreateLayout('Layout #2')


# set active view
SetActiveView(None)

# Create a new 'Histogram View'
histogramView1 = CreateView('XYHistogramChartView')
histogramView1.ViewSize = [400, 200]
histogramView1.LeftAxisRangeMaximum = 6.66
histogramView1.BottomAxisRangeMaximum = 6.66
histogramView1.ChartTitle = label
histogramView1.BottomAxisTitle = "Mesh quality"
histogramView1.LeftAxisUseCustomRange = 1
histogramView1.LeftAxisRangeMaximum = hist_maximum
histogramView1.LeftAxisRangeMinimum = 0.0
# get layout
layout2 = GetLayout()

# place view in the layout
layout2.AssignView(0, histogramView1)

# set active source
SetActiveSource(meshQuality1)

# show data in view
meshQuality1Display_1 = Show(meshQuality1, histogramView1)
# trace defaults for the display properties.
meshQuality1Display_1.SelectInputArray = ['CELLS', 'Quality']
meshQuality1Display_1.UseCustomBinRanges = 1
meshQuality1Display_1.CustomBinRanges = [1+1e-5, 1.5-1e-5]

# Properties modified on meshQuality1Display_1
meshQuality1Display_1.BinCount = 40

### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.0, 10000.0]
renderView1.CameraParallelScale = 1.0

# animationScene.GoToLast()
ExportView(img_directory + label + "_mesh_quality_final.eps", view=histogramView1)
animationScene.GoToFirst()
histogramView1.ChartTitle = "Initial mesh quality"
ExportView(img_directory + label + "_mesh_quality_initial.eps", view=histogramView1)

def get_quality_array(meshQuality1):
    fetch = servermanager.Fetch(meshQuality1)
    rcCellData = fetch.GetCellData()
    quality_array = rcCellData.GetArray("Quality")
    quality_array = [quality_array.GetValue(i) for i in range(fetch.GetNumberOfCells())]
    quality_array = np.asarray(quality_array)
    x = np.sort(quality_array)
    return x

xfirst = get_quality_array(meshQuality1)
animationScene.GoToLast()
xlast = get_quality_array(meshQuality1)
y = np.asarray(list(range(len(xfirst))))/float(len(xfirst))
np.save(img_directory + "inv_cdf_" + label, np.column_stack((xfirst, xlast, y)))
