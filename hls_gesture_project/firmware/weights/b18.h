//Numpy array shape [5]
//Min -0.031250000000
//Max 0.023437500000
//Number of zeros 1

#ifndef B18_H_
#define B18_H_

#ifndef __SYNTHESIS__
bias18_t b18[5];
#else
bias18_t b18[5] = {0.0156250, -0.0234375, 0.0000000, 0.0234375, -0.0312500};
#endif

#endif
