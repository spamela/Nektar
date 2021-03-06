// --- Execute as
// gmsh iter_with_square.geo -2 -o iter_grid.msh
// gmsh iter_with_square.geo -2 -o iter_grid.vtk


// --- length scale of grid
lc1 = 0.4;
lc2 = 0.1;

// limiter points
Point(1)  = { 4.2    ,  -2.4    , 0.0000000, lc1 };
Point(2)  = { 4.3257 ,  -2.6514 , 0.0000000, lc1 };
Point(3)  = { 4.4408 ,  -2.7808 , 0.0000000, lc1 };
Point(4)  = { 4.5066 ,  -2.941  , 0.0000000, lc1 };
Point(5)  = { 4.5157 ,  -3.1139 , 0.0000000, lc1 };
Point(6)  = { 4.467  ,  -3.2801 , 0.0000000, lc1 };
Point(7)  = { 4.3773 ,  -3.4689 , 0.0000000, lc1 };
Point(8)  = { 4.3115 ,  -3.6075 , 0.0000000, lc1 };
Point(9)  = { 4.2457 ,  -3.7461 , 0.0000000, lc1 };
Point(10) = { 4.1799 ,  -3.8847 , 0.0000000, lc1 };
Point(11) = { 4.4918 ,  -3.9092 , 0.0000000, lc1 };
Point(12) = { 4.5687 ,  -3.8276 , 0.0000000, lc1 };
Point(13) = { 4.6456 ,  -3.7460 , 0.0000000, lc1 };
Point(14) = { 4.8215 ,  -3.7090 , 0.0000000, lc1 };
Point(15) = { 4.9982 ,  -3.7414 , 0.0000000, lc1 };
Point(16) = { 5.1496 ,  -3.8382 , 0.0000000, lc1 };
Point(17) = { 5.2529 ,  -3.9852 , 0.0000000, lc1 };
Point(18) = { 5.2628 ,  -4.1244 , 0.0000000, lc1 };
Point(19) = { 5.2727 ,  -4.2636 , 0.0000000, lc1 };
Point(20) = { 5.565  ,  -4.5559 , 0.0000000, lc1 };
Point(21) = { 5.565  ,  -4.4026 , 0.0000000, lc1 };
Point(22) = { 5.565  ,  -4.2494 , 0.0000000, lc1 };
Point(23) = { 5.565  ,  -4.0962 , 0.0000000, lc1 };
Point(24) = { 5.565  ,  -3.9961 , 0.0000000, lc1 };
Point(25) = { 5.572  ,  -3.896  , 0.0000000, lc1 };
Point(26) = { 5.6008 ,  -3.7024 , 0.0000000, lc1 };
Point(27) = { 5.6842 ,  -3.5265 , 0.0000000, lc1 };
Point(28) = { 5.815  ,  -3.3823 , 0.0000000, lc1 };
Point(29) = { 5.9821 ,  -3.2822 , 0.0000000, lc1 };
Point(30) = { 6.2176 ,  -3.2288 , 0.0000000, lc1 };
Point(31) = { 6.267  ,  -3.046  , 0.0000000, lc1 };
Point(32) = { 7.283  ,  -2.257  , 0.0000000, lc1 };
Point(33) = { 7.899  ,  -1.342  , 0.0000000, lc1 };
Point(34) = { 8.28   ,  -0.421  , 0.0000000, lc1 };
Point(35) = { 8.395  ,   0.58   , 0.0000000, lc1 };
Point(36) = { 8.395  ,   0.72   , 0.0000000, lc1 };
Point(37) = { 8.27   ,   1.681  , 0.0000000, lc1 };
Point(38) = { 7.904  ,   2.464  , 0.0000000, lc1 };
Point(39) = { 7.4    ,   3.179  , 0.0000000, lc1 };
Point(40) = { 6.587  ,   3.894  , 0.0000000, lc1 };
Point(41) = { 5.753  ,   4.532  , 0.0000000, lc1 };
Point(42) = { 4.904  ,   4.712  , 0.0000000, lc1 };
Point(43) = { 4.311  ,   4.324  , 0.0000000, lc1 };
Point(44) = { 4.126  ,   3.582  , 0.0000000, lc1 };
Point(45) = { 4.076  ,   2.566  , 0.0000000, lc1 };
Point(46) = { 4.046  ,   1.549  , 0.0000000, lc1 };
Point(47) = { 4.046  ,   0.533  , 0.0000000, lc1 };
Point(48) = { 4.067  ,   -0.484 , 0.0000000, lc1 };
Point(49) = { 4.097  ,   -1.5   , 0.0000000, lc1 };
Point(50) = { 4.15   ,   -2.1   , 0.0000000, lc1 };

// --- separatrix points
Point(51)  = { 0.819596856E+01 , 0.323997695E+00  ,   0.0000000, lc2 };
Point(52)  = { 0.819997994E+01 , 0.537505869E+00  ,   0.0000000, lc2 };
Point(53)  = { 0.819062963E+01 , 0.750391378E+00  ,   0.0000000, lc2 };
Point(54)  = { 0.816930566E+01 , 0.953062367E+00  ,   0.0000000, lc2 };
Point(55)  = { 0.813808522E+01 , 0.114708016E+01  ,   0.0000000, lc2 };
Point(56)  = { 0.809757409E+01 , 0.133349654E+01  ,   0.0000000, lc2 };
Point(57)  = { 0.804837934E+01 , 0.151304993E+01  ,   0.0000000, lc2 };
Point(58)  = { 0.799160641E+01 , 0.168664269E+01  ,   0.0000000, lc2 };
Point(59)  = { 0.792742935E+01 , 0.185456211E+01  ,   0.0000000, lc2 };
Point(60)  = { 0.785618378E+01 , 0.201699869E+01  ,   0.0000000, lc2 };
Point(61)  = { 0.777842470E+01 , 0.217425160E+01  ,   0.0000000, lc2 };
Point(62)  = { 0.769437959E+01 , 0.232620441E+01  ,   0.0000000, lc2 };
Point(63)  = { 0.760449820E+01 , 0.247287844E+01  ,   0.0000000, lc2 };
Point(64)  = { 0.750884175E+01 , 0.261354639E+01  ,   0.0000000, lc2 };
Point(65)  = { 0.740823571E+01 , 0.274858360E+01  ,   0.0000000, lc2 };
Point(66)  = { 0.730295526E+01 , 0.287733757E+01  ,   0.0000000, lc2 };
Point(67)  = { 0.719355350E+01 , 0.299951425E+01  ,   0.0000000, lc2 };
Point(68)  = { 0.708067075E+01 , 0.311499861E+01  ,   0.0000000, lc2 };
Point(69)  = { 0.696494273E+01 , 0.322369255E+01  ,   0.0000000, lc2 };
Point(70)  = { 0.684698315E+01 , 0.332542661E+01  ,   0.0000000, lc2 };
Point(71)  = { 0.672740195E+01 , 0.341995598E+01  ,   0.0000000, lc2 };
Point(72)  = { 0.660685514E+01 , 0.350737831E+01  ,   0.0000000, lc2 };
Point(73)  = { 0.648595366E+01 , 0.358770769E+01  ,   0.0000000, lc2 };
Point(74)  = { 0.636528506E+01 , 0.366049212E+01  ,   0.0000000, lc2 };
Point(75)  = { 0.623901026E+01 , 0.372941886E+01  ,   0.0000000, lc2 };
Point(76)  = { 0.610021675E+01 , 0.379626673E+01  ,   0.0000000, lc2 };
Point(77)  = { 0.594811474E+01 , 0.385818135E+01  ,   0.0000000, lc2 };
Point(78)  = { 0.578234033E+01 , 0.391141117E+01  ,   0.0000000, lc2 };
Point(79)  = { 0.560367710E+01 , 0.394915437E+01  ,   0.0000000, lc2 };
Point(80)  = { 0.541481082E+01 , 0.396200892E+01  ,   0.0000000, lc2 };
Point(81)  = { 0.522197006E+01 , 0.393598274E+01  ,   0.0000000, lc2 };
Point(82)  = { 0.503517443E+01 , 0.385662066E+01  ,   0.0000000, lc2 };
Point(83)  = { 0.486610743E+01 , 0.371579460E+01  ,   0.0000000, lc2 };
Point(84)  = { 0.472251574E+01 , 0.351982910E+01  ,   0.0000000, lc2 };
Point(85)  = { 0.460577108E+01 , 0.328547151E+01  ,   0.0000000, lc2 };
Point(86)  = { 0.451286709E+01 , 0.303049434E+01  ,   0.0000000, lc2 };
Point(87)  = { 0.443941943E+01 , 0.276810806E+01  ,   0.0000000, lc2 };
Point(88)  = { 0.438159041E+01 , 0.250602456E+01  ,   0.0000000, lc2 };
Point(89)  = { 0.433626111E+01 , 0.224819206E+01  ,   0.0000000, lc2 };
Point(90)  = { 0.430120697E+01 , 0.199595574E+01  ,   0.0000000, lc2 };
Point(91)  = { 0.427457400E+01 , 0.174931010E+01  ,   0.0000000, lc2 };
Point(92)  = { 0.425540168E+01 , 0.150700199E+01  ,   0.0000000, lc2 };
Point(93)  = { 0.424289126E+01 , 0.126728734E+01  ,   0.0000000, lc2 };
Point(94)  = { 0.423648766E+01 , 0.102787477E+01  ,   0.0000000, lc2 };
Point(95)  = { 0.423602477E+01 , 0.785836331E+00  ,   0.0000000, lc2 };
Point(96)  = { 0.424175734E+01 , 0.537505869E+00  ,   0.0000000, lc2 };
Point(97)  = { 0.425364749E+01 , 0.291230702E+00  ,   0.0000000, lc2 };
Point(98)  = { 0.427090549E+01 , 0.550651240E-01  ,   0.0000000, lc2 };
Point(99)  = { 0.429324807E+01 ,-0.174960496E+00  ,   0.0000000, lc2 };
Point(100) = { 0.432044785E+01 ,-0.402101396E+00  ,   0.0000000, lc2 };
Point(101) = { 0.435245941E+01 ,-0.629154932E+00  ,   0.0000000, lc2 };
Point(102) = { 0.438986810E+01 ,-0.858297234E+00  ,   0.0000000, lc2 };
Point(103) = { 0.443277864E+01 ,-0.109180562E+01  ,   0.0000000, lc2 };
Point(104) = { 0.448183446E+01 ,-0.133153568E+01  ,   0.0000000, lc2 };
Point(105) = { 0.453755024E+01 ,-0.157943793E+01  ,   0.0000000, lc2 };
Point(106) = { 0.460094434E+01 ,-0.183694803E+01  ,   0.0000000, lc2 };
Point(107) = { 0.467308825E+01 ,-0.210532546E+01  ,   0.0000000, lc2 };
Point(108) = { 0.475590078E+01 ,-0.238420935E+01  ,   0.0000000, lc2 };
Point(109) = { 0.485258576E+01 ,-0.266944908E+01  ,   0.0000000, lc2 };
Point(110) = { 0.496846697E+01 ,-0.294806856E+01  ,   0.0000000, lc2 };
Point(111) = { 0.511696383E+01 ,-0.317309953E+01  ,   0.0000000, lc2 };
Point(112) = { 0.532056526E+01 ,-0.322655844E+01  ,   0.0000000, lc2 };
Point(113) = { 0.553972251E+01 ,-0.316062934E+01  ,   0.0000000, lc2 };
Point(114) = { 0.574218837E+01 ,-0.306878666E+01  ,   0.0000000, lc2 };
Point(115) = { 0.592434904E+01 ,-0.297234455E+01  ,   0.0000000, lc2 };
Point(116) = { 0.608756039E+01 ,-0.287685271E+01  ,   0.0000000, lc2 };
Point(117) = { 0.623387258E+01 ,-0.278427489E+01  ,   0.0000000, lc2 };
Point(118) = { 0.636528506E+01 ,-0.269572308E+01  ,   0.0000000, lc2 };
Point(119) = { 0.648967039E+01 ,-0.260664573E+01  ,   0.0000000, lc2 };
Point(120) = { 0.661341386E+01 ,-0.251299975E+01  ,   0.0000000, lc2 };
Point(121) = { 0.673611859E+01 ,-0.241432870E+01  ,   0.0000000, lc2 };
Point(122) = { 0.685735362E+01 ,-0.231043598E+01  ,   0.0000000, lc2 };
Point(123) = { 0.697667781E+01 ,-0.220124851E+01  ,   0.0000000, lc2 };
Point(124) = { 0.709362654E+01 ,-0.208666583E+01  ,   0.0000000, lc2 };
Point(125) = { 0.720768291E+01 ,-0.196650184E+01  ,   0.0000000, lc2 };
Point(126) = { 0.731827553E+01 ,-0.184055554E+01  ,   0.0000000, lc2 };
Point(127) = { 0.742480022E+01 ,-0.170868898E+01  ,   0.0000000, lc2 };
Point(128) = { 0.752666519E+01 ,-0.157089175E+01  ,   0.0000000, lc2 };
Point(129) = { 0.762349683E+01 ,-0.142753830E+01  ,   0.0000000, lc2 };
Point(130) = { 0.771453465E+01 ,-0.127831739E+01  ,   0.0000000, lc2 };
Point(131) = { 0.779948756E+01 ,-0.112363558E+01  ,   0.0000000, lc2 };
Point(132) = { 0.787767057E+01 ,-0.963309357E+00  ,   0.0000000, lc2 };
Point(133) = { 0.794871580E+01 ,-0.797497139E+00  ,   0.0000000, lc2 };
Point(134) = { 0.801234360E+01 ,-0.626283580E+00  ,   0.0000000, lc2 };
Point(135) = { 0.806751731E+01 ,-0.449130819E+00  ,   0.0000000, lc2 };
Point(136) = { 0.811465075E+01 ,-0.266331570E+00  ,   0.0000000, lc2 };
Point(137) = { 0.815186287E+01 ,-0.768058449E-01  ,   0.0000000, lc2 };
Point(138) = { 0.817943904E+01 , 0.119615146E+00  ,   0.0000000, lc2 };

// --- points for square in the middle
Point(139) = { 5.0 , 1.0 , 0.0000000, lc1 };
Point(140) = { 7.0 , 1.0 , 0.0000000, lc1 };
Point(141) = { 7.0 ,-1.0 , 0.0000000, lc1 };
Point(142) = { 5.0 ,-1.0 , 0.0000000, lc1 };

// --- straight curves between points for wall
Line(1)  = {1 , 2 };
Line(2)  = {2 , 3 };
Line(3)  = {3 , 4 };
Line(4)  = {4 , 5 };
Line(5)  = {5 , 6 };
Line(6)  = {6 , 7 };
Line(7)  = {7 , 8 };
Line(8)  = {8 , 9 };
Line(9)  = {9 , 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 17};
Line(17) = {17, 18};
Line(18) = {18, 19};
Line(19) = {19, 20};
Line(20) = {20, 21};
Line(21) = {21, 22};
Line(22) = {22, 23};
Line(23) = {23, 24};
Line(24) = {24, 25};
Line(25) = {25, 26};
Line(26) = {26, 27};
Line(27) = {27, 28};
Line(28) = {28, 29};
Line(29) = {29, 30};
Line(30) = {30, 31};
Line(31) = {31, 32};
Line(32) = {32, 33};
Line(33) = {33, 34};
Line(34) = {34, 35};
Line(35) = {35, 36};
Line(36) = {36, 37};
Line(37) = {37, 38};
Line(38) = {38, 39};
Line(39) = {39, 40};
Line(40) = {40, 41};
Line(41) = {41, 42};
Line(42) = {42, 43};
Line(43) = {43, 44};
Line(44) = {44, 45};
Line(45) = {45, 46};
Line(46) = {46, 47};
Line(47) = {47, 48};
Line(48) = {48, 49};
Line(49) = {49, 50};
Line(50) = {50, 1 };

// --- straight curves between points for separatrix
Line(51)  = {51 ,52  };
Line(52)  = {52 ,53  };
Line(53)  = {53 ,54  };
Line(54)  = {54 ,55  };
Line(55)  = {55 ,56  };
Line(56)  = {56 ,57  };
Line(57)  = {57 ,58  };
Line(58)  = {58 ,59  };
Line(59)  = {59 ,60  };
Line(60)  = {60 ,61  };
Line(61)  = {61 ,62  };
Line(62)  = {62 ,63  };
Line(63)  = {63 ,64  };
Line(64)  = {64 ,65  };
Line(65)  = {65 ,66  };
Line(66)  = {66 ,67  };
Line(67)  = {67 ,68  };
Line(68)  = {68 ,69  };
Line(69)  = {69 ,70  };
Line(70)  = {70 ,71  };
Line(71)  = {71 ,72  };
Line(72)  = {72 ,73  };
Line(73)  = {73 ,74  };
Line(74)  = {74 ,75  };
Line(75)  = {75 ,76  };
Line(76)  = {76 ,77  };
Line(77)  = {77 ,78  };
Line(78)  = {78 ,79  };
Line(79)  = {79 ,80  };
Line(80)  = {80 ,81  };
Line(81)  = {81 ,82  };
Line(82)  = {82 ,83  };
Line(83)  = {83 ,84  };
Line(84)  = {84 ,85  };
Line(85)  = {85 ,86  };
Line(86)  = {86 ,87  };
Line(87)  = {87 ,88  };
Line(88)  = {88 ,89  };
Line(89)  = {89 ,90  };
Line(90)  = {90 ,91  };
Line(91)  = {91 ,92  };
Line(92)  = {92 ,93  };
Line(93)  = {93 ,94  };
Line(94)  = {94 ,95  };
Line(95)  = {95 ,96  };
Line(96)  = {96 ,97  };
Line(97)  = {97 ,98  };
Line(98)  = {98 ,99  };
Line(99)  = {99 ,100 };
Line(100) = {100,101 };
Line(101) = {101,102 };
Line(102) = {102,103 };
Line(103) = {103,104 };
Line(104) = {104,105 };
Line(105) = {105,106 };
Line(106) = {106,107 };
Line(107) = {107,108 };
Line(108) = {108,109 };
Line(109) = {109,110 };
Line(110) = {110,111 };
Line(111) = {111,112 };
Line(112) = {112,113 };
Line(113) = {113,114 };
Line(114) = {114,115 };
Line(115) = {115,116 };
Line(116) = {116,117 };
Line(117) = {117,118 };
Line(118) = {118,119 };
Line(119) = {119,120 };
Line(120) = {120,121 };
Line(121) = {121,122 };
Line(122) = {122,123 };
Line(123) = {123,124 };
Line(124) = {124,125 };
Line(125) = {125,126 };
Line(126) = {126,127 };
Line(127) = {127,128 };
Line(128) = {128,129 };
Line(129) = {129,130 };
Line(130) = {130,131 };
Line(131) = {131,132 };
Line(132) = {132,133 };
Line(133) = {133,134 };
Line(134) = {134,135 };
Line(135) = {135,136 };
Line(136) = {136,137 };
Line(137) = {137,138 };
Line(138) = {138,51  };

// --- straight curves between points in centre square
Line(139) = {139 , 140 };
Line(140) = {140 , 141 };
Line(141) = {141 , 142 };
Line(142) = {142 , 139 };

// --- contour with all lines for wall
Curve Loop(1) = 
{1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
26,
27,
28,
29,
30,
31,
32,
33,
34,
35,
36,
37,
38,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
50};

// --- contour with all lines for separatrix
Curve Loop(2) = 
{51,
52,
53,
54,
55,
56,
57,
58,
59,
60,
61,
62,
63,
64,
65,
66,
67,
68,
69,
70,
71,
72,
73,
74,
75,
76,
77,
78,
79,
80,
81,
82,
83,
84,
85,
86,
87,
88,
89,
90,
91,
92,
93,
94,
95,
96,
97,
98,
99,
100,
101,
102,
103,
104,
105,
106,
107,
108,
109,
110,
111,
112,
113,
114,
115,
116,
117,
118,
119,
120,
121,
122,
123,
124,
125,
126,
127,
128,
129,
130,
131,
132,
133,
134,
135,
136,
137,
138};

// --- contour with all lines for centre square
Curve Loop(3) = {139, 140, 141, 142};

// --- 2D surface inside contour wall
Plane Surface(1) = {1,2};

// --- 2D surface inside contour separatrix
Plane Surface(2) = {2,3};


//Physical Surface("wall") = {1};
//Physical Surface("separatrix") = {2};

//Curve(2) In Surface(1);














// At this level, Gmsh knows everything to display the rectangular surface 1 and
// to mesh it. An optional step is needed if we want to group elementary
// geometrical entities into more meaningful groups, e.g. to define some
// mathematical ("domain", "boundary"), functional ("left wing", "fuselage") or
// material ("steel", "carbon") properties.
//
// Such groups are called "Physical Groups" in Gmsh. By default, if physical
// groups are defined, Gmsh will export in output files only mesh elements that
// belong to at least one physical group. (To force Gmsh to save all elements,
// whether they belong to physical groups or not, set `Mesh.SaveAll=1;', or
// specify `-save_all' on the command line.) Physical groups are also identified
// by tags, i.e. strictly positive integers, that should be unique per dimension
// (0D, 1D, 2D or 3D). Physical groups can also be given names.
//
// Here we define a physical curve that groups the left, bottom and right curves
// in a single group (with prescribed tag 5); and a physical surface with name
// "My surface" (with an automatic tag) containing the geometrical surface 1:

//Physical Curve(5) = {1, 2, 4};
//Physical Surface("My surface") = {1};

// Now that the geometry is complete, you can
// - either open this file with Gmsh and select `2D' in the `Mesh' module to
//   create a mesh; then select `Save' to save it to disk in the default format
//   (or use `File->Export' to export in other formats);
// - or run `gmsh t1.geo -2` to mesh in batch mode on the command line.

// You could also uncomment the following lines in this script:
//
//   Mesh 2;
//   Save "t1.msh";
//
// which would lead Gmsh to mesh and save the mesh every time the file is
// parsed. (To simply parse the file from the command line, you can use `gmsh
// t1.geo -')

// By default, Gmsh saves meshes in the latest version of the Gmsh mesh file
// format (the `MSH' format). You can save meshes in other mesh formats by
// specifying a filename with a different extension in the GUI, on the command
// line or in scripts. For example
//
//   Save "t1.unv";
//
// will save the mesh in the UNV format. You can also save the mesh in older
// versions of the MSH format:
//
// - In the GUI: open `File->Export', enter your `filename.msh' and then pick
//   the version in the dropdown menu.
// - On the command line: use the `-format' option (e.g. `gmsh file.geo -format
//   msh2 -2').
// - In a `.geo' script: add `Mesh.MshFileVersion = x.y;' for any version
//   number `x.y'.
// - As an alternative method, you can also not specify the format explicitly,
//   and just choose a filename with the `.msh2' or `.msh4' extension.

// Note that starting with Gmsh 3.0, models can be built using other geometry
// kernels than the default built-in kernel. By specifying
//
//   SetFactory("OpenCASCADE");
//
// any subsequent command in the `.geo' file would be handled by the OpenCASCADE
// geometry kernel instead of the built-in kernel. Different geometry kernels
// have different features. With OpenCASCADE, instead of defining the surface by
// successively defining 4 points, 4 curves and 1 curve loop, one can define the
// rectangular surface directly with
//
//   Rectangle(2) = {.2, 0, 0, .1, .3};
//
// The underlying curves and points could be accessed with the `Boundary' or
// `CombinedBoundary' operators.
//
// See e.g. `t16.geo', `t18.geo', `t19.geo' or `t20.geo' for complete examples
// based on OpenCASCADE, and `demos/boolean' for more.


