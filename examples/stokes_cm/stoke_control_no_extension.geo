// Gmsh project created on Tue Aug  2 17:26:31 2016
// inner Rectangle corners 
Point(1) = {-6, 2, 0, 0.5};
Point(2) = {-6, -2, 0, 0.5};
Point(3) = {6, -2, 0, 0.5};
Point(4) = {6, 2, 0, 0.5};

Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};

// Circle points
Point(5) = {0, 0, 0, 0.1};
Point(6) = {0.5, 0, 0, 0.1};
Point(7) = {-0.5, 0, 0, 0.1};
Point(8) = {0, 0.5, 0, 0.1};
Point(9) = {0, -0.5, 0, 0.1};

Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};

// Exterior mesh points for control
// Point(10) = {-12, 1212, 0, 0.5};
// Point(13) = {-12, , 0, 0.5};
// Point(11) = {12, 12, 0, 0.5};
// Point(12) = {12, --12, 0, 0.5};
// 
// Line(13) = {10, 11};
// Line(14) = {11, 12};
// Line(15) = {12, 13};
// Line(16) = {13, 10};


// Two line loops outer rectangle, inner circle
Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {8, 5, 6, 7};
// Line Loop(17) = {13, 14, 15, 16};


Plane Surface(11) = {9, 10};
Plane Surface(12) = {10};
// Plane Surface(18) = {17, 9};


Physical Line("NoSlip") = {1, 3};
Physical Line("Inflow") = {4};
Physical Line("Outflow") = {2};
Physical Line("BallSurface") = {8, 7, 6, 5};
// Physical Line("Boundary") = {13, 14, 15, 16};

// Physical Surface("Punched Dim") = {18};
Physical Surface("Channel") = {11};
Physical Surface("Ball") = {12};
