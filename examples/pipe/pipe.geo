h = 1.; //width
H = 5.; //height

//lower points
Point(1) = { 0, 0, 0, 1.};
Point(2) = { 2, 0, 0, 1.}; //first spline point
Point(3) = { 4, 0, 0, 1.};
Point(4) = { 8, 6, 0, 1.};
Point(5) = {10, 6, 0, 1.};
Point(6) = {12, 6, 0, 1.}; //last spline point
Point(7) = {15, 6, 0, 1.};
//upper points
Point( 8) = {15, 7, 0, 1.};
Point( 9) = {12, 7, 0, 1.}; //first spline point
Point(10) = {10, 7, 0, 1.};
Point(11) = { 8, 7, 0, 1.};
Point(12) = { 4, 1, 0, 1.};
Point(13) = { 2, 1, 0, 1.}; //last spline point
Point(14) = { 0, 1, 0, 1.};

//edges
Line(1) = {1, 2};
BSpline(2) = {2, 3, 4, 5, 6};
Line(3) = {6, 7};
Line(4) = {7, 8};
Line(5) = {8, 9};
BSpline(6) = {9, 10, 11, 12, 13};
Line(7) = {13, 14};
Line(8) = {14,  1};

//boundary and physical curves
Curve Loop(9) = {1, 2, 3, 4, 5, 6, 7, 8};
Physical Curve("Inflow", 10) = {8};
Physical Curve("Outflow", 11) = {4};
Physical Curve("WallFixed", 12) = {1, 3, 5, 7};
Physical Curve("WallFree", 13) = {2, 6};


//domain and physical surface
Plane Surface(1) = {9};
Physical Surface("Pipe", 2) = {1};
