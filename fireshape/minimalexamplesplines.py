from firedrake import *
from fireshape import *
import ipdb

N = 30; Order = 3; Level = 6;
#N = 80; Order = 3; Level = 6;

mesh = UnitCubeMesh(N, N, N)

import datetime as dt
t_ = dt.datetime.now()
print (t_.strftime("%Y-%m-%d %H:%M:%S"), " assemble FeControlSpace", flush=True)
Q = FeControlSpace(mesh)
print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
       " assemble Fecontrolspace finished, it took : ", dt.datetime.now() - t_, flush=True),

t_ = dt.datetime.now()
print (t_.strftime("%Y-%m-%d %H:%M:%S"), " assemble BsplineControlSpace", flush=True)
bbox = [(-0.1, 1.2), (-0.2, 1.3), (-0.01, 1.)]
orders = [Order, Order, Order]
levels = [Level, Level, Level]
Q = BsplineControlSpace(mesh, bbox, orders, levels)#,
print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
       " assemble BsplineControlSpace finished, it took : ", dt.datetime.now() - t_, flush=True),
                        #boundary_regularities=[2, 1, 0])
t_ = dt.datetime.now()
print (t_.strftime("%Y-%m-%d %H:%M:%S"), " assemble H1InnerProduct", flush=True)
inner = H1InnerProduct(Q)
print (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
       " assemble H1InnerProduct finished, it took : ", dt.datetime.now() - t_, flush=True),
#A = Q.FullIFW
#print(A.getSize())
#ipdb.set_trace()
