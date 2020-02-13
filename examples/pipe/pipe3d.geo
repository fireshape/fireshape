// Gmsh project created on Tue Jan 22 11:40:52 2019
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 0.5, 0, 2*Pi};
Line Loop(2) = {1};
Plane Surface(1) = {2};
Point( 5) = {0, 0.0,  0.00, 1.0};
Point( 6) = {0, 0.0,  2.00, 1.0}; //first spline point
Point( 7) = {0, 0.0,  3.00, 1.0};
Point( 8) = {0, 0.0,  4.00, 1.0};
Point( 9) = {0, 0.2,  4.75, 1.0};
Point(10) = {0, 5.0,  6.00, 1.0};
Point(11) = {0, 5.0, 10.00, 1.0};
Point(12) = {0, 5.0, 12.00, 1.0}; //last spline point
Point(13) = {0, 5.0, 15.00, 1.0};

Bezier(10) = {6, 7, 8, 9, 10, 11, 12};
Line(15) = {5, 6};
Line(16) = {12, 13};
Wire(1) = {15, 10, 16};
Extrude { Surface{1}; } Using Wire {1}
Delete{ Surface{1}; }

// When using fancy commands like "extrude", it is not quite obvious
// what the numbering of the generated lines, surfaces and volumes is
// going to be. To figure this out, we open the .geo file in the gmsh
// gui and use the Tools -> Visibility menu to find the numbers for
// each entity.

Physical Surface("Inflow", 10) = {2};
Physical Surface("Outflow", 11) = {6};
Physical Surface("WallFixed", 12) = {3, 5};
Physical Surface("WallFree", 13) = {4};
//Physical Surface("Inflow") = {2};
//Physical Surface("Outflow") = {6};
//Physical Surface("WallFixed") = {3, 5};
//Physical Surface("WallFree") = {4};
Physical Volume("PhysVol") = {1};
