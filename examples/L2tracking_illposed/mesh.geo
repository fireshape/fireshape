SetFactory("OpenCASCADE");

//external points
Point(1) = {-4, -2, 0, 1.0};
Point(2) = {4, -2, 0, 1.0};
Point(3) = {4, 2, 0, 1.0};
Point(4) = {-4, 2, 0, 1.0};

//edges
Circle(1) = {-2, 0, 0, 0.5, 0, 2*Pi};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line(5) = {4, 1};

//boundary and physical curves
Curve Loop(1) = {1};
Curve Loop(2) = {2, 3, 4, 5};
Physical Curve("inner_boundary") = {1};
Physical Curve("external_boundary") = {2, 3, 4, 5};

//domain and physical surface
Plane Surface(1) = {1, 2};
Physical Surface("domain") = {1};
