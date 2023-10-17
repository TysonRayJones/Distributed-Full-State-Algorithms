#ifndef MISC_HPP
#define MISC_HPP


#include <algorithm>

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

        #pragma omp parallel for
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


static NatArray getReorderedAllSuffixTargets(NatArray targets, Nat suffixSize) {

    // locate largest non-targeted qubit of the suffix subregister
    Index targetMask = getBitMask(targets);
    Nat maxSuffixNonTarg = getNextLeftmostZeroBit(targetMask, suffixSize);
        
    // determine desired target ordering, where prefix targets are swapped with largest un-targeted un-swapped suffix qubits
    NatArray reorderedTargets(0);
    for (Nat q = targets.size(); q-- != 0; ) {
        if (targets[q] < suffixSize)
            reorderedTargets.push_back(targets[q]);
        else {
            reorderedTargets.push_back(maxSuffixNonTarg);
            maxSuffixNonTarg = getNextLeftmostZeroBit(targetMask, maxSuffixNonTarg);
        }
    }
    std::reverse(reorderedTargets.begin(), reorderedTargets.end());

    return reorderedTargets;
}


static NatArray getNonTargetedQubitOrder(Nat numAllQubits, NatArray originalTargets, NatArray reorderedTargets) {

    // determine the ordering of all qubits after swaps
    NatArray allQubits(numAllQubits);
    for (Nat q=0; q<allQubits.size(); q++)
        allQubits[q] = q;
    for (Nat q=0; q<reorderedTargets.size(); q++) {
        Nat qb1 = originalTargets[q];
        Nat qb2 = reorderedTargets[q];
        if (qb1 != qb2)
            std::swap(allQubits[qb1], allQubits[qb2]);
    }

    // remove the targeted qubit indices, momentarily maintaining non-targeted indices
    NatArray remainingQubits(0);
    Index reorderedMask = getBitMask(reorderedTargets);
    for (Nat i=0; i<allQubits.size(); i++)
        if (!getBit(reorderedMask, i))
            remainingQubits.push_back(allQubits[i]);

    // shift remaining qubits to be contiguous [0, ..., 2N-len(reorderedTargets))
    Index remainingMask = getBitMask(remainingQubits);
    for (Nat &q : remainingQubits) {
        Nat p = q;
        for (Nat i=0; i<p; i++)
            q -= ! getBit(remainingMask, i);
    }

    return remainingQubits;
}



#endif // MISC_HPP