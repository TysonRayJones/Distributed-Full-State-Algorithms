#ifndef BIT_MATHS_HPP
#define BIT_MATHS_HPP


#include "types.hpp"



/* 
 * Fast and inlined; safe to call in tight loops.
 * None of these twiddles may use sign tricks, 
 * since Nat and Index are unsigned.
 */


#define INLINE static inline __attribute__((always_inline))


INLINE Index powerOf2(Nat exponent) {
    
    return 1ULL << exponent;
}


INLINE bool isPowerOf2(Index number) {

    return (number > 0) && ((number & (number - 1U)) == 0);
}


INLINE Nat getBit(Index number, Nat bitIndex) {
    
    return (number >> bitIndex) & 1U;
}


INLINE Index flipBit(Index number, Nat bitIndex) {
    
    return number ^ (1ULL << bitIndex);
}


INLINE Index insertBit(Index number, Nat bitIndex, Nat bitValue) {
    
    Index left = (number >> bitIndex) << (bitIndex + 1);
    Index middle = bitValue << bitIndex;
    Index right = number & ((1ULL << bitIndex) - 1);
    return left | middle | right;
}


INLINE Index insertBits(Index number, NatArray bitIndices, Nat bitValue) {
    
    // bitIndices must be strictly increasing
    for (Nat i: bitIndices)
        number = insertBit(number, i, bitValue);
        
    return number;
}


INLINE Index setBit(Index number, Nat bitIndex, Nat bitValue) {
    
    Index mask = bitValue << bitIndex;
    return (number & (~mask)) | mask;
}


INLINE Index setBits(Index number, NatArray bitIndices, Index bitsValue) {
    
    for (Nat i=0; i<bitIndices.size(); i++) {
        Nat bit = getBit(bitsValue, i);
        number = setBit(number, bitIndices[i], bit);
    }
    
    return number;
}


INLINE Nat getBitMaskParity(Index mask) {
    
    // BEWARE:
    //  - this compiler extension may not be available on all compilers
    //  - the (Nat) cast may add additional clock cycles
    
    return (Nat) __builtin_parity(mask);
}



/*
 * Aliases for code clarity 
 */
 

 INLINE Index insertTwoBits(Index number, Nat highInd, Nat highBit, Nat lowInd, Nat lowBit) {
     
     number = insertBit(number, lowInd, lowBit);
     number = insertBit(number, highInd, highBit);
     return number;
 }
 
 
 INLINE Index insertThreeZeroBits(Index number, Nat i3, Nat i2, Nat i1) {
     
     number = insertTwoBits(number, i2, 0, i1, 0);
     number = insertBit(number, i3, 0);
     return number;
 }


 INLINE Index insertFourZeroBits(Index number, Nat i4, Nat i3, Nat i2, Nat i1) {
     
     number = insertTwoBits(number, i2, 0, i1, 0);
     number = insertTwoBits(number, i4, 0, i3, 0);
     return number;
 }
 
 INLINE Index flipTwoBits(Index number, Nat i1, Nat i0) {
     
     number = flipBit(number, i1);
     number = flipBit(number, i0);
     return number;
 }



 /* 
 * Non-performance critical code; not to be used in tight loops
 */


static Nat getNextLeftmostZeroBit(Index mask, Nat bitInd) {

    bitInd--;
    while (getBit(mask, bitInd))
        bitInd--;
    
    return bitInd;
}


static bool allBitsAreOne(Index number, NatArray bitIndices) {
    
    for (Nat i : bitIndices)
        if (!getBit(number, i))
            return false;
            
    return true;
}


static Index getBitMask(NatArray bitIndices) {
    
    Index mask = 0;
    for (Nat i: bitIndices)
        mask = flipBit(mask, i);
        
    return mask;
}


static Nat logBase2(Index powerOf2) {
    
    Nat expo = 0;
    while (getBit(powerOf2, 0) != 1) {
        expo++;
        powerOf2 >>= 1;
    }

    return expo;
}


#endif // BIT_MATHS_HPP