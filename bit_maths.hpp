
#ifndef BIT_MATHS_HPP
#define BIT_MATHS_HPP

#include "types.hpp"
#include "structures.hpp"



/* 
 * fast and inlined; safe to call in tight loops
 */

#define INLINE static inline __attribute__((always_inline))

INLINE Index powerOf2(Nat exponent) {
    
    return 1ULL << exponent;
}

INLINE Nat getBit(Nat number, Nat bitIndex) {
    
    return (number >> bitIndex) & 1U;
}

INLINE Nat flipBit(Nat number, Nat bitIndex) {
    
    return number ^ (1U << bitIndex);
}



/* 
 * slow; do not call in tight loops 
 */

Nat logBase2(Nat powerOf2) {
    
    int expo = 0;
    while (getBit(powerOf2, 0) != 1) {
        expo++;
        powerOf2 >>= 1;
    }
    
    return expo;
}



#endif // BIT_MATHS_HPP