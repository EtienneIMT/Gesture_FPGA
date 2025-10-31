//Numpy array shape [16]
//Min -128.000000000000
//Max 127.000000000000
//Number of zeros 0

#ifndef B96_H_
#define B96_H_

#ifndef __SYNTHESIS__
bias96_t b96[16];
#else
bias96_t b96[16] = {127, -128, 127, -128, -128, 127, -128, -128, 127, 127, -128, -128, 127, -128, 127, 127};

#endif

#endif
