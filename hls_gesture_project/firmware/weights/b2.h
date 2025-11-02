//Numpy array shape [8]
//Min -0.015625000000
//Max 0.000000000000
//Number of zeros 2

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[8];
#else
bias2_t b2[8] = {-0.0078125, 0.0000000, -0.0156250, 0.0000000, -0.0078125, -0.0078125, -0.0156250, -0.0078125};
#endif

#endif
