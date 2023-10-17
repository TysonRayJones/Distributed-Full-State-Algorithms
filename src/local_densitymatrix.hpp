#ifndef LOCAL_DENSITYMATRIX_HPP
#define LOCAL_DENSITYMATRIX_HPP


#include "types.hpp"
#include "states.hpp"
#include "bit_maths.hpp"

#include <algorithm>


static void local_densitymatrix_oneQubitDephasing(DensityMatrix &rho, Nat qb, Real prob) {

    Amp fac = 1 - 2*prob;

    if (qb >= rho.numQubits - rho.logNumNodes) {
        
        Index numIts = rho.numAmpsPerNode / 2;
        Nat bit = ! getBit(rho.rank, qb - (rho.numQubits - rho.logNumNodes));
        
        #pragma omp parallel for
        for (Index k=0; k<numIts; k++) {
            Index j = insertBit(k, qb, bit);
            rho.amps[j] *= fac;
        }
    }
    
    else {
        
        Index numIts = rho.numAmpsPerNode / 4;
        Nat altQb = qb + rho.numQubits;
        
        #pragma omp parallel for
        for (Index k=0; k<numIts; k++) {
            Index j01 = insertTwoBits(k, altQb, 0, qb, 1);
            rho.amps[j01] *= fac;
            
            Index j10 = insertTwoBits(k, altQb, 1, qb, 0);
            rho.amps[j10] *= fac;
        }
    }
}


static void local_densitymatrix_twoQubitDephasing(DensityMatrix &rho, Nat qb1, Nat qb2, Real prob) {
    
    Index rankShift = rho.rank << rho.logNumAmpsPerNode;
    Nat alt1 = qb1 + rho.numQubits;
    Nat alt2 = qb2 + rho.numQubits;
    Amp term = - 4*prob/3;
    
    #pragma omp parallel for
    for (Index j=0; j<rho.numAmpsPerNode; j++) {
        Index i = rankShift | j;
        Nat bit1 = getBit(i, qb1) ^ getBit(i, alt1);
        Nat bit2 = getBit(i, qb2) ^ getBit(i, alt2);
        Amp flag = Amp(bit1 | bit2, 0);
        rho.amps[j] *= flag * term + 1.;
    }
}


static void local_densitymatrix_oneQubitDepolarising(DensityMatrix &rho, Nat qb, Amp c1, Amp c2, Amp c3) {
    
    Index numIts = rho.numAmpsPerNode / 4;
    Nat altQb = qb + rho.numQubits;
    
    #pragma omp parallel for
    for (Index k=0; k<numIts; k++) {
        Index j00 = insertTwoBits(k, altQb, 0, qb, 0);
        Index j01 = flipBit(j00, qb);
        Index j10 = flipBit(j00, altQb);
        Index j11 = flipBit(j01, altQb);
        
        Amp amp00 = rho.amps[j00];
        rho.amps[j00] = c2*amp00 + c1*rho.amps[j11];
        rho.amps[j01] *= c3;
        rho.amps[j10] *= c3;
        rho.amps[j11] = c1*amp00 + c2*rho.amps[j11];
    }
}

static void local_densitymatrix_twoQubitDepolarising(DensityMatrix &rho, Nat q0, Nat q1, Nat q2, Nat q3, Amp c1, Amp c2, Amp c3) {
    
    #pragma omp parallel for
    for (Index j=0; j<rho.numAmpsPerNode; j++) {
        Nat flag1 = !(getBit(j, q0) ^ getBit(j, q2));
        Nat flag2 = !(getBit(j, q1) ^ getBit(j, q3));
        Amp fac = 1. + c3 * Amp(!(flag1 & flag2), 0);
        rho.amps[j] *= fac;
    }
    
    Index numIts = rho.numAmpsPerNode / 16;

    #pragma omp parallel for
    for (Index k=0; k<numIts; k++) {
        Index j0000 = insertFourZeroBits(k, q3, q2, q1, q0);
        Index j0101 = flipTwoBits(j0000, q2, q0);
        Index j1010 = flipTwoBits(j0000, q3, q1);
        Index j1111 = flipTwoBits(j0101, q3, q1);
        
        Amp term = rho.amps[j0000] + rho.amps[j0101] + rho.amps[j1010] + rho.amps[j1111];
        rho.amps[j0000] = c1*rho.amps[j0000] + c2*term;
        rho.amps[j0101] = c1*rho.amps[j0101] + c2*term;
        rho.amps[j1010] = c1*rho.amps[j1010] + c2*term;
        rho.amps[j1111] = c1*rho.amps[j1111] + c2*term;
    }
}


static void local_densitymatrix_damping(DensityMatrix &rho, Nat qb, Real prob) {
    
    Nat qbAlt = qb + rho.numQubits;
    Amp c1 = sqrt(1 - prob);
    Amp c2 = 1 - prob;

    Index numIts = rho.numAmpsPerNode / 4;

    #pragma omp parallel for
    for (Index k=0; k<numIts; k++) {
        Index j00 = insertTwoBits(k, qbAlt, 0, qb, 0);
        Index j01 = flipBit(j00, qb);
        Index j10 = flipBit(j00, qbAlt);
        Index j11 = flipBit(j01, qbAlt);
        
        rho.amps[j00] += prob*rho.amps[j11];
        rho.amps[j01] *= c1;
        rho.amps[j10] *= c1;
        rho.amps[j11] *= c2;
    }
}


static DensityMatrix local_densitymatrix_partialTrace(DensityMatrix &inRho, NatArray targs, NatArray pairTargs) {

    DensityMatrix outRho = DensityMatrix(inRho.numQubits - targs.size());

    // sort all targets
    NatArray allTargs = targs;
    allTargs.insert(allTargs.end(), pairTargs.begin(), pairTargs.end());
    std::sort(allTargs.begin(), allTargs.end());

    Index numTracedAmps = powerOf2(targs.size());

    #pragma omp parallel for
    for (Index l=0; l<outRho.numAmpsPerNode; l++) {

        outRho.amps[l] = 0.;

        Index lMask = insertBits(l, allTargs, 0);

        for (Index k=0; k<numTracedAmps; k++) {

            Index i = lMask;
            i = setBits(i, targs, k);
            i = setBits(i, pairTargs, k);

            outRho.amps[l] += inRho.amps[i];
        }
    }

    // return-value optimisation in most compilers should avoid a copy here
    return outRho;
}


#endif // LOCAL_DENSITYMATRIX_HPP