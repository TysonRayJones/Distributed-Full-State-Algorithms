#ifndef DISTRIBUTED_DENSITYMATRIX_HPP
#define DISTRIBUTED_DENSITYMATRIX_HPP

#include "types.hpp"
#include "states.hpp"
#include "bit_maths.hpp"
#include "misc.hpp"
#include "communication.hpp"
#include "local_densitymatrix.hpp"



static void distributed_densitymatrix_manyTargGate(DensityMatrix &rho, NatArray targets, AmpMatrix gate) {
    
    distributed_statevector_manyTargGate(rho, targets, gate);
    
    for (Nat &t : targets)
        t += rho.numQubits;
        
    gate = getConjugateMatrix(gate);
    distributed_statevector_manyTargGate(rho, targets, gate);
}


static void distributed_densitymatrix_swapGate(DensityMatrix &rho, Nat qb1, Nat qb2) {
    
    Nat n = rho.numQubits;
    distributed_statevector_swapGate(rho, qb1, qb2);
    distributed_statevector_swapGate(rho, qb1 + n, qb2 + n);
}


static void distributed_densitymatrix_pauliTensor(DensityMatrix &rho, NatArray targets, NatArray paulis) {
    
    distributed_statevector_pauliTensor(rho, targets, paulis);
    
    for (Nat &t : targets)
        t += rho.numQubits;
    
    distributed_statevector_pauliTensor(rho, targets, paulis);
    
    if (containsOddNumY(paulis))
        for (Amp &amp : rho.amps)
            amp *= -1;
}


static void distributed_densitymatrix_pauliGadget(DensityMatrix &rho, NatArray targets, NatArray paulis, Real theta) {

    distributed_statevector_pauliGadget(rho, targets, paulis, theta);
    
    for (Nat &t : targets)
        t += rho.numQubits;
        
    if (!containsOddNumY(paulis))
        theta *= -1;

    distributed_statevector_pauliGadget(rho, targets, paulis, theta);
}


static void distributed_densitymatrix_phaseGadget(DensityMatrix &rho, NatArray targets, Real theta) {
    
    distributed_statevector_phaseGadget(rho, targets, theta);
    
    for (Nat &t : targets)
        t += rho.numQubits;
        
    theta *= -1;
    distributed_statevector_phaseGadget(rho, targets, theta);
}


static void distributed_densitymatrix_krausMap(DensityMatrix &rho, MatrixArray krausOps, NatArray targets) {
    
    AmpMatrix superOp = getSuperoperator(krausOps);

    for (Nat t : targets)
        targets.push_back(t + rho.numQubits);

    distributed_statevector_manyTargGate(rho, targets, superOp); 
}


static void distributed_densitymatrix_oneQubitDephasing(DensityMatrix &rho, Nat qb, Real prob) {
    
    local_densitymatrix_oneQubitDephasing(rho, qb, prob);
}


static void distributed_densitymatrix_twoQubitDephasing(DensityMatrix &rho, Nat qb1, Nat qb2, Real prob) {
    
    local_densitymatrix_twoQubitDephasing(rho, qb1, qb2, prob);
}


static void distributed_densitymatrix_oneQubitDepolarising(DensityMatrix &rho, Nat qb, Real prob) {
    
    Amp c1 = 2*prob/3;
    Amp c2 = 1 - 2*prob/3;
    Amp c3 = 1 - 4*prob/3;
    
    if (qb < rho.numQubits - rho.logNumNodes)
        local_densitymatrix_oneQubitDepolarising(rho, qb, c1, c2, c3);
    
    else {
        Index numIts = rho.numAmpsPerNode / 2;
        Nat qbShift = qb - rho.numQubits + rho.logNumNodes;
        Nat bit = getBit(rho.rank, qbShift);
        
        // pack half of local amps into buffer
        for (Index k=0; k<numIts; k++) {
            Index j = insertBit(k, qb, ! bit);
            rho.buffer[k] = rho.amps[j];
        }
        
        // swap half-buffers
        Nat pairRank = flipBit(rho.rank, qbShift);
        comm_exchangeArrays(rho.buffer, 0, rho.buffer, 0, numIts, pairRank);
        
        for (Index k=0; k<numIts; k++) {
            Index j = insertBit(k, qb, ! bit);
            rho.amps[j] *= c3;
        }
        
        for (Index k=0; k<numIts; k++) {
            Index j = insertBit(k, qb, bit);
            rho.amps[j] = c2*rho.amps[j] + c1*rho.buffer[k];
        }
    }
}


static void distributed_densitymatrix_twoQubitDepolarising_subroutine_pair(DensityMatrix &rho, Nat q0, Nat q1, Nat q2, Amp c1, Amp c2, Amp c3) {

    Nat alt1 = q1 - (rho.numQubits - rho.logNumNodes);
    Nat bit = getBit(rho.rank, alt1);
    
    // scale all amplitudes
    for (Index j=0; j<rho.numAmpsPerNode; j++) {
        Nat flag1 = getBit(j, q0) == getBit(j, q2); 
        Nat flag2 = getBit(j, q1) == bit;
        Amp fac = 1. + c3 * Amp(!(flag1 & flag2), 0);
        rho.amps[j] *= fac;
    }
    
    // pack eighth of buffer with pre-summed amp pairs
    Index numIts = rho.numAmpsPerNode / 8;
    for (Index k=0; k<numIts; k++) {
        Index j000 = insertThreeZeroBits(k, q2, q1, q0);
        Index j0b0 = setBit(j000, q1, bit);
        Index j1b1 = flipTwoBits(j0b0, q2, q0);
        rho.buffer[k] = rho.amps[j0b0] + rho.amps[j1b1];
    }
    
    // swap eighth-buffers
    Nat pairRank = flipBit(rho.rank, alt1);
    comm_exchangeArrays(rho.buffer, 0, rho.buffer, 0, numIts, pairRank);
    
    for (Index k=0; k<numIts; k++) {
        Index j000 = insertThreeZeroBits(k, q2, q1, q0);
        Index j0b0 = setBit(j000, q1, bit);
        Index j1b1 = flipTwoBits(j0b0, q2, q0);
        rho.amps[j0b0] = c1*rho.amps[j0b0] + c2*(rho.amps[j1b1] + rho.buffer[k]);
        rho.amps[j1b1] = c1*rho.amps[j1b1] + c2*(rho.amps[j0b0] + rho.buffer[k]);
    }
}


static void distributed_densitymatrix_twoQubitDepolarising_subroutine_quad(DensityMatrix &rho, Nat q0, Nat q1, Amp c1, Amp c2, Amp c3) {

    Nat alt0 = q0 - (rho.numQubits - rho.logNumNodes);
    Nat alt1 = q1 - (rho.numQubits - rho.logNumNodes);
    Nat bit0 = getBit(rho.rank, alt0);
    Nat bit1 = getBit(rho.rank, alt1);
    
    // scale all amplitudes
    for (Index j=0; j<rho.numAmpsPerNode; j++) {
        Nat flag1 = getBit(j, q0) == bit0; 
        Nat flag2 = getBit(j, q1) == bit1;
        Amp fac = 1. + c3 * Amp(!(flag1 & flag2), 0);
        rho.amps[j] *= fac;
    }
    
    // pack fourth of buffer
    Index numIts = rho.numAmpsPerNode / 4;
    for (Index k=0; k<numIts; k++) {
        Index j = insertTwoBits(k, q1, bit1, q0, bit0);
        rho.buffer[k] = rho.amps[j];
    }
    
    // swap fourth-buffer with first pair node
    Nat pairRank0 = flipBit(rho.rank, alt0);
    comm_exchangeArrays(rho.buffer, 0, rho.buffer, 0, numIts, pairRank0);
    
    // update local amps and buffer
    for (Index k=0; k<numIts; k++) {
        Index j = insertTwoBits(k, q1, bit1, q0, bit0);
        Amp amp = c1*rho.amps[j] + c2*rho.buffer[k];
        rho.amps[j] = amp;
        rho.buffer[k] = amp;
    }
    
    // swap fourth-buffer with second pair node
    Nat pairRank1 = flipBit(rho.rank, alt1);
    comm_exchangeArrays(rho.buffer, 0, rho.buffer, 0, numIts, pairRank1);
    
    // update local amps 
    Amp c4 = c2/c1;
    for (Index k=0; k<numIts; k++) {
        Index j = insertTwoBits(k, q1, bit1, q0, bit0);
        rho.amps[j] = c4 * rho.buffer[k];
    }
}


static void distributed_densitymatrix_twoQubitDepolarising(DensityMatrix &rho, Nat qb1, Nat qb2, Real prob) {

    // ensure qb2 is larger
    if (qb1 > qb2)
        std::swap(qb1, qb2);
        
    Amp c1 = 1 - 4*prob/5;
    Amp c2 = 4*prob/15;
    Amp c3 = -16*prob/15;
    Nat shift1 = qb1 + rho.numQubits;
    Nat shift2 = qb2 + rho.numQubits;
    
    Nat threshold = rho.numQubits - rho.logNumNodes;
        
    if (qb2 < threshold)
        local_densitymatrix_twoQubitDepolarising(rho, qb1, qb2, shift1, shift2, c1, c2, c3);
        
    else if (qb2 >= threshold && qb1 < threshold)
        distributed_densitymatrix_twoQubitDepolarising_subroutine_pair(rho, qb1, qb2, shift1, c1, c2, c3);
    
    else
        distributed_densitymatrix_twoQubitDepolarising_subroutine_quad(rho, qb1, qb2, c1, c2, c3);
}


static void distributed_densitymatrix_damping(DensityMatrix &rho, Nat qb, Real prob) {
    
    Nat threshold = rho.numQubits - rho.logNumNodes;
    
    if (qb < threshold)
        local_densitymatrix_damping(rho, qb, prob);
        
    else {
        Index numIts = rho.numAmpsPerNode / 2;
        Nat pairRank = flipBit(rho.rank, qb - threshold);
        Nat bit = getBit(rho.rank, qb - threshold);
        Amp c1 = sqrt(1 - prob);
        Amp c2 = 1 - prob;
        
        // half of all nodes...
        if (bit == 1) {
            
            // pack half buffer and scale half their amps
            for (Index k=0; k<numIts; k++) {
                Index j = insertBit(k, qb, 1);
                rho.buffer[k] = rho.amps[j];
                rho.amps[j] *= c2;
            }
            
            // asynchronously send half-buffer
            comm_asynchSendArray(rho.buffer, numIts, pairRank);
        }
        
        // all nodes proceed to scale half their (remaining) local amps
        for (Index k=0; k<numIts; k++) {
            Index j = insertBit(k, qb, !bit);
            rho.amps[j] *= c1;
        }
        
        // other half of all nodes...
        if (bit == 0) {
            
            // receive half-buffer 
            comm_receiveArray(rho.buffer, numIts, pairRank);
            
            // and combine it with their remaining local amps
            for (Index k=0; k<numIts; k++) {
                Index j = insertBit(k, qb, 0);
                rho.amps[j] += prob * rho.buffer[k];
            }
        }
        
        // stop asynch senders from proceeding to maintain buffer integrity
        comm_synch();
    }
}


static Amp distributed_densitymatrix_expecPauliString(DensityMatrix &rho, RealArray coeffs, NatArray allPaulis) {

    Amp value = 0;
    
    for (Index j=0; j<rho.numAmpsPerNode; j++) {
        Index i = (rho.rank << rho.logNumAmpsPerNode) | j;

        Amp term = 0;
        
        for (Index t=0; t<coeffs.size(); t++) {
            Nat* termPaulis = &allPaulis[t*rho.numQubits];
            Amp elem = getPauliTensorElem(termPaulis, rho.numQubits, i);
            term += elem * coeffs[t];
        }
        
        value += term * rho.amps[j];
    }
    
    comm_reduceAmp(value);
    return value;
}



#endif // DISTRIBUTED_DENSITYMATRIX_HPP