// Gmsh project created on Tue Jan 22 11:40:52 2019
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 0.5, 0, 2*Pi};
Line Loop(2) = {1};
Plane Surface(1) = {2};
Point(5) = {0, 0, 0, 1.0};
Point(6) = {0, 0, 1.5, 1.0};
Point(7) = {0, 0, 3.0, 1.0};
Point(8) = {0, 0, 4.0, 1.0};
Point(9) = {0, 0.5, 4.75, 1.0};
Point(10) = {0, 5.0, 5.5, 1.0};
Point(11) = {0, 5.0, 6.5, 1.0};
Point(12) = {0, 5.0, 10.0, 1.0};
Point(13) = {0, 5.0, 16.0, 1.0};


Bezier(10) = {6, 7, 8, 9, 10, 11, 12};
Line(15) = {5, 6};
Line(16) = {12, 13};
Wire(1) = {15, 10, 16};
Extrude { Surface{1}; } Using Wire {1}
Delete{ Surface{1}; }
Physical Surface("Inflow") = {2};
Physical Surface("Outflow") = {6};
Physical Surface("WallFixed") = {3, 5};
Physical Surface("WallFree") = {4};
Physical Volume("PhysVol") = {1};
