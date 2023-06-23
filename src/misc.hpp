
#ifndef MISC_HPP
#define MISC_HPP

#include "bit_maths.hpp"
#include "types.hpp"



/* 
 * Relatively fast compared to caller; can be used in tight-loops
 */

INLINE Amp getPauliTensorElem(Nat* pauliCodes, Nat numQubits, Index flatInd) {
    
    AmpMatrix I{{1,0},{0,1}};
    AmpMatrix X{{0,1},{1,0}};
    AmpMatrix Y{{0,Amp(0,-1)},{Amp(0,1),0}};
    AmpMatrix Z{{1,0},{0,-1}};
    MatrixArray pauliMatrices{ I, X, Y, Z };
    
    Amp elem = 1.;
    
    for (Nat q=0; q<numQubits; q++) {
        Nat code = pauliCodes[q];
        AmpMatrix matrix = pauliMatrices[code];
        
        Nat colBit = getBit(flatInd, q);
        Nat rowBit = getBit(flatInd, q + numQubits);
    
        Amp fac = matrix[rowBit][colBit];
        elem *= fac;
    }

    return elem;
}



/* 
 * Slow; not to be used in tight-loops
 */


static Nat logBase2(Index powerOf2) {
    
    Nat expo = 0;
    while (getBit(powerOf2, 0) != 1) {
        expo++;
        powerOf2 >>= 1;
    }

    return expo;
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


static bool containsOddNumY(NatArray paulis) {
    
    bool isOddNumY = false;
    for (Nat op : paulis)
        if ((PauliOperator) op == Y)
            isOddNumY = ! isOddNumY;
            
    return isOddNumY;
}


static AmpMatrix getSuperoperator(MatrixArray krausOps) {
    
    Index krausDim = krausOps[0].size();
    Index superDim = krausDim * krausDim;
    AmpMatrix superOp = getZeroMatrix(superDim);
    
    for (AmpMatrix krausOp : krausOps) {
        
        // superOp += conj(krausOp) (tensor) krausOp
        for (Index i=0; i<krausDim; i++)
            for (Index j=0; j<krausDim; j++)
                for (Index k=0; k<krausDim; k++)
                    for (Index l=0; l<krausDim; l++) {
                        Index r = i*krausDim + k;
                        Index c = j*krausDim + l;
                        Amp term = conj(krausOp[i][j]) * krausOp[k][l];
                        superOp[r][c] += term;
                    }
    }
    
    return superOp;        
}



#endif // MISC_HPP