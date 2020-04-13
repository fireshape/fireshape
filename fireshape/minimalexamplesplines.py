from firedrake import *
from fireshape import *
import ipdb

#N = 2; Order = 2; Level = 2;
N = 10; Order = 3; Level = 3;

mesh = UnitCubeMesh(N, N, N)
bbox = [(-0.1, 1.2), (-0.2, 1.3), (-0.01, 1.)]
orders = [Order, Order, Order]
levels = [Level, Level, Level]
#Q = FeControlSpace(mesh)
Q = BsplineControlSpace(mesh, bbox, orders, levels)#,
                        #boundary_regularities=[2, 1, 0])
#inner = H1InnerProduct(Q)
#A = Q.FullIFW
#print(A.getSize())
#ipdb.set_trace()
