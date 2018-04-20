// Gmsh project created on Tue Aug  2 17:26:31 2016
Point(1) = {0., 1.0, 0, 0.5};
Point(2) = {0.,  .0, 0, 0.5};
Point(3) = {1.0, .0, 0, 0.5};
Point(4) = {1.0, 1.0, 0, 0.5};

Point(5) = {0.5, 0.5, 0, 0.1};
Point(6) = {0.8, 0.5, 0, 0.1};
Point(7) = {0.2, 0.5, 0, 0.1};
Point(8) = {0.5, 0.75, 0, 0.1};
Point(9) = {0.5, 0.25, 0, 0.1};
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};
Ellipse(5) = {7, 5, 6, 8};
Ellipse(6) = {8, 5, 9, 6};
Ellipse(7) = {6, 5, 7, 9};
Ellipse(8) = {9, 5, 8, 7};
Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {8, 5, 6, 7};
Plane Surface(11) = {9, 10};
Physical Line("Outer") = {1,2,3,4};
Physical Line("Inner") = {8, 7, 6, 5};
Physical Surface("Channel") = {11};
