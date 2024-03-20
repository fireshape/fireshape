 // -----------------------------------------------------------------------------
//
//  Gmsh GEO tutorial 4
//
//  Built-in functions, holes in surfaces, annotations, entity colors
//
// -----------------------------------------------------------------------------

lc = 0.01;

// Exterior corners of rectangle
Point(1) = {-2, -2, 0, lc};
Point(2) = {3, -2, 0, lc};
Point(3) = {3, 3, 0, lc};
Point(4) = {-2, 3, 0, lc};

// points to define inner rectangle
Point(5) = {0, 0, 0, lc};
Point(6) = {1, 0, 0, lc};
Point(7) = {1, 1, 0, lc};
Point(8) = {0, 1, 0, lc};

// exterior rectangle lines
Line(1) = {1, 2}; 
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// exterior rectangle curve loop
Curve Loop(5) = {1, 2, 3, 4};

// circle curves need 2 as arc must be less than pi
Line(6) = {5, 6}; 
Line(7) = {6, 7};
Line(8) = {7, 8};
Line(9) = {8, 5};

Curve Loop(10) = {6, 7, 8, 9};

// plane surface for interior square
Plane Surface(1) = {10};

// plane surface for exterior rectangle with interior circle hole
Plane Surface(2) = {5, 10};

Physical Curve("HorEdges", 9) = {1, 3};
Physical Curve("VerEdges", 10) = {2, 4};
Physical Curve("Interior Edges", 11) = {6, 7, 8, 9};

Physical Surface("PunchedDom", 3) = {2};
Physical Surface("Disc", 4) = {1};