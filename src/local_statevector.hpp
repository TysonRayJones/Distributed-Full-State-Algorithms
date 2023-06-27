
#ifndef LOCAL_STATEVECTOR_HPP
#define LOCAL_STATEVECTOR_HPP


#include "types.hpp"
#include "states.hpp"
#include "bit_maths.hpp"

#include <algorithm>
#include <cmath>


static void local_statevector_oneTargGate(StateVector& psi, Nat target, AmpMatrix gate) {
    
    Index numIts = psi.numAmpsPerNode / 2;
    
    for (Index j=0; j<numIts; j++) {
        Index i0 = insertBit(j, target, 0);
        Index i1 = flipBit(i0, target);
        
        Amp amp0 = psi.amps[i0];
        Amp amp1 = psi.amps[i1];
        
        psi.amps[i0] = gate[0][0]*amp0 + gate[0][1]*amp1;
        psi.amps[i1] = gate[1][0]*amp0 + gate[1][1]*amp1;
    }
}


static void local_statevector_manyCtrlOneTargGate(StateVector& psi, NatArray controls, Nat target, AmpMatrix gate) {
    
    NatArray qubits = controls;
    qubits.push_back(target);
    std::sort(qubits.begin(), qubits.end());
    
    Index numIts = psi.numAmpsPerNode / powerOf2(qubits.size());
    
    for (Index j=0; j<numIts; j++) {
        Index i1 = insertBits(j, qubits, 1);
        Index i0 = flipBit(i1, target);
        
        Amp amp0 = psi.amps[i0];
        Amp amp1 = psi.amps[i1];
        
        psi.amps[i0] = gate[0][0]*amp0 + gate[0][1]*amp1;
        psi.amps[i1] = gate[1][0]*amp0 + gate[1][1]*amp1;
    }
}


static void local_statevector_manyTargGate(StateVector &psi, NatArray targets, AmpMatrix gate) {
    
    AmpArray cache(gate.size());
    NatArray qubits = targets;
    std::sort(qubits.begin(), qubits.end());
    
    Index numIts = psi.numAmpsPerNode / gate.size();
    for (Index k=0; k<numIts; k++) {
        
        Index baseInd = insertBits(k, qubits, 0);
        
        for (Index j=0; j<gate.size(); j++) {
            Index i = setBits(baseInd, targets, j);
            cache[j] = psi.amps[i];
        }
        
        for (Index j=0; j<gate.size(); j++) {
            Index i = setBits(baseInd, targets, j);
            psi.amps[i] = 0;
            for (Index l=0; l<gate.size(); l++)
                psi.amps[i] += gate[j][l] * cache[l];
        }
    }
}


static void local_statevector_pauliTensorOrGadget_subroutine(StateVector &psi, NatArray targs, Amp powI, Index maskXY, Index maskYZ, Amp thisAmpFac, Amp otherAmpFac) {
    
    // targs must be incresaing order before subsequent calls of insertBits()
    std::sort(targs.begin(), targs.end());

    Index numOuterIts = psi.numAmpsPerNode / powerOf2(targs.size());
    Index numInnerIts = powerOf2(targs.size()) / 2;
    Index rankShift = psi.rank << psi.logNumAmpsPerNode;
    
    for (Index k=0; k<numOuterIts; k++) {
        Index h = insertBits(k, targs, 0);
        
        for (Index l=0; l<numInnerIts; l++) {
            
            // determine indices and signs of amps swapped by X and Y
            Index j0 = setBits(h, targs, l);
            Index i0 = rankShift | j0;
            Nat p0 = getBitMaskParity(i0 & maskYZ);
            Amp b0 = (1. - 2.*p0) * powI;
            
            Index j1 = j0 ^ maskXY;
            Index i1 = rankShift | j1;
            Nat p1 = getBitMaskParity(i1 & maskYZ);
            Amp b1 = (1. - 2.*p1) * powI;
            
            // mix scaled amps (swaps scaled amps if theta=)
            Amp amp0 = psi.amps[j0];
            Amp amp1 = psi.amps[j1];
            psi.amps[j0] = thisAmpFac*amp0 + otherAmpFac*b1*amp1;
            psi.amps[j1] = thisAmpFac*amp1 + otherAmpFac*b0*amp0;
        }
    }
}


static void local_statevector_phaseGadget(StateVector &psi, NatArray targets, Real theta) {
        
    Index rankShift = psi.rank << psi.logNumAmpsPerNode;
    Index targMask = getBitMask(targets);
    
    Amp fac0 = Amp(cos(theta), +sin(theta)); // exp(+i theta)
    Amp fac1 = Amp(cos(theta), -sin(theta)); // exp(-i theta)
    AmpArray facs = {fac0, fac1};
    
    for (Index j=0; j<psi.numAmpsPerNode; j++) {
        Index i = rankShift | j;
        Nat p = getBitMaskParity(i & targMask);
        psi.amps[j] *= facs[p]; // fac1*p + fac0*(!p);
    }
}


#endif // LOCAL_STATEVECTOR_HPP