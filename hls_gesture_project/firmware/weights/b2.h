//Numpy array shape [8]
//Min -0.062500000000
//Max 0.070312500000
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
bias2_t b2[8];
#else
bias2_t b2[8] = {0.0078125, 0.0078125, 0.0078125, 0.0234375, -0.0625000, 0.0703125, 0.0156250, -0.0156250};
#endif

#endif
