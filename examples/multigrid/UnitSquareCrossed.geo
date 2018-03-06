// Gmsh project created on Tue Jan 23 10:52:17 2018
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {-0, 1, 0, 1.0};
//+
Line(1) = {1, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 1};
//+
Line Loop(6) = {4, 5, 2, 3};
//+
Plane Surface(7) = {6};
//+
Physical Surface("Interior") = {7};
