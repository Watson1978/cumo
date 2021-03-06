#ifndef CUMO_SCOMPLEX_KERNEL_H
#define CUMO_SCOMPLEX_KERNEL_H

typedef cumo_scomplex dtype;
typedef float rtype;

#include "complex_macro_kernel.h"

__device__ static inline bool c_nearly_eq(dtype x, dtype y) {
    return c_abs(c_sub(x,y)) <= (c_abs(x)+c_abs(y))*FLT_EPSILON*2;
}

#endif // CUMO_SCOMPLEX_KERNEL_H
