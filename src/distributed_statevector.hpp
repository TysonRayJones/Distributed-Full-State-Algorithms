
#ifndef DISTRIBUTED_STATEVECTOR_HPP
#define DISTRIBUTED_STATEVECTOR_HPP

#include "types.hpp"
#include "states.hpp"
#include "bit_maths.hpp"
#include "misc.hpp"
#include "communication.hpp"

#include "local_statevector.hpp"

#include <cmath>
#include <algorithm>



void distributed_statevector_oneTargGate(StateVector& psi, Nat target, AmpMatrix gate) {
    
    // embarrassingly parallel
    if (target < psi.logNumAmpsPerNode)
        local_statevector_oneTargGate(psi, target, gate);
    
    else {
        // exchange all amps (receive to buffer)
        Nat rankTarget = target - psi.logNumAmpsPerNode;
        Nat pairRank = flipBit(psi.rank, rankTarget);
        comm_exchangeArrays(psi.amps, psi.buffer, pairRank);
        
        // extract relevant gate elements
        Nat bit = getBit(psi.rank, rankTarget);
        Amp fac0 = gate[bit][bit];
        Amp fac1 = gate[bit][!bit];
        
        // update psi using local and received amps
        for (Index i=0; i<psi.numAmpsPerNode; i++)
            psi.amps[i] = fac0*psi.amps[i] + fac1*psi.buffer[i];
    }
}


static void distributed_statevector_manyCtrlOneTargGate_subroutine(StateVector& psi, NatArray controls, Nat target, AmpMatrix gate) {
    
    // controls must be sorted for subsequent insertBits() calls
    std::sort(controls.begin(), controls.end());
        
    Nat localTarget = target - psi.logNumAmpsPerNode;
    Nat pairRank = flipBit(psi.rank, localTarget);
    Nat bufferOffset = 0;

    // because controls.size() > 0, it's gauranteed that numAmpsToMod < logNumAmpsPerNode/2
    Index numAmpsToMod = psi.numAmpsPerNode / powerOf2(controls.size()); 

    // pack sub-buffer[0...]
    for (Index j=0; j<numAmpsToMod; j++) {
        Index k = insertBits(j, controls, 1);
        psi.buffer[j] = psi.amps[k];
    }

    // send buffer[0...], receive buffer[bufferOffset...] (gauranteed to fit)
    bufferOffset = numAmpsToMod;
    comm_exchangeArrays(psi.buffer, 0, psi.buffer, bufferOffset, numAmpsToMod, pairRank);
    
    // extract relevant gate elements
    Nat bit = getBit(psi.rank, localTarget);
    Amp fac0 = gate[bit][bit];
    Amp fac1 = gate[bit][!bit];
    
    // update psi using sub-buffer
    for (Index j=0; j<numAmpsToMod; j++) {
        Index k = insertBits(j, controls, 1);
        Index l = j + bufferOffset;
        psi.amps[k] = fac0*psi.amps[k] + fac1*psi.buffer[l];
    }
}


static void distributed_statevector_manyCtrlOneTargGate(StateVector& psi, NatArray controls, Nat target, AmpMatrix gate) {
    
    NatArray prefixCtrls = NatArray(0);
    NatArray suffixCtrls = NatArray(0);
    for(Nat q : controls)
        if (q >= psi.logNumAmpsPerNode)
            prefixCtrls.push_back(q - psi.logNumAmpsPerNode);
        else
            suffixCtrls.push_back(q);
            
    // do nothing if this node fails prefix control condition
    if (!allBitsAreOne(psi.rank, prefixCtrls))
        return;
    
    // embarrassingly parallel
    if (target < psi.logNumAmpsPerNode)
        local_statevector_manyCtrlOneTargGate(psi, suffixCtrls, target, gate);
    
    // no suffix controls; effect non-controlled gate
    else if (suffixCtrls.size() == 0)
        distributed_statevector_oneTargGate(psi, target, gate);
    
    // bespoke communication for controls required
    else
        distributed_statevector_manyCtrlOneTargGate_subroutine(psi, suffixCtrls, target, gate);
}


static void distributed_statevector_swapGate(StateVector &psi, Nat qb1, Nat qb2) {
    
    // ensure qb2 is larger
    if (qb1 > qb2)
        std::swap(qb1, qb2);
        
    // embarrassingly parallel
    if (qb2 < psi.logNumAmpsPerNode) {
        Index numIts = psi.numAmpsPerNode / 4;
        for (Index k=0; k<numIts; k++) {
            Index j11 = insertTwoBits(k, qb2, 1, qb1, 1);
            Index j10 = flipBit(j11, qb1);
            Index j01 = flipBit(j11, qb2);
            std::swap(psi.amps[j01], psi.amps[j10]);
        }
    }
    
    // zero or one full-statevector swaps
    else if (qb1 >= psi.logNumAmpsPerNode) {
        Nat alt1 = qb1 - psi.logNumAmpsPerNode;
        Nat alt2 = qb2 - psi.logNumAmpsPerNode;
        if (getBit(psi.rank, alt1) != getBit(psi.rank, alt2)) {
            
            Nat pairRank = flipBit(psi.rank, alt1);
            pairRank = flipBit(pairRank, alt2);

            // directly swap amps (although MPI arrays must not overlap)
            comm_exchangeArrays(psi.amps, psi.buffer, pairRank);

            for (Index j=0; j<psi.numAmpsPerNode; j++)
                psi.amps[j] = psi.buffer[j];
        }
    }
    
    // communicate and overwrite half of local amps...
    else {
        Nat alt2 = qb2 - psi.logNumAmpsPerNode;
        Nat pairRank = flipBit(psi.rank, alt2);
        Index bitMask = ( ! getBit(psi.rank, alt2) ) << qb1;
        
        // via direct overwrites (if msgSize sufficiently large)
        if (qb1 >= 30) {
            Index msgSize = powerOf2(qb1);
            Index numIts = psi.numAmpsPerNode / msgSize;
            for (Index k=0; k<numIts; k++) {
                Index i = (k * msgSize * 2) | bitMask;
                
                // directly swap amps subset (although MPI arrays must not overlap)
                comm_exchangeArrays(psi.amps, i, psi.buffer, i, msgSize, pairRank);

                for (Index j=0; j<msgSize; j++) {
                    Index l = i + j;
                    psi.amps[l] = psi.buffer[l];
                }
            }
            
        // else via a wasteful full swap
        } else {
            comm_exchangeArrays(psi.amps, psi.buffer, pairRank);
            
            Index numIts = psi.numAmpsPerNode / 2;
            for (Index k=0; k<numIts; k++) {
                Index j = insertBit(k, qb1, 0);
                Index i = (j & (~bitMask)) | bitMask;
                Index l = flipBit(i, qb1);
                psi.amps[i] = psi.buffer[l];
            }
        }
    }
}


static void distributed_statevector_manyTargGate(StateVector &psi, NatArray targets, AmpMatrix gate) {
    assert( targets.size() <= psi.logNumAmpsPerNode );
    
    // locate smallest non-targeted qubit
    Index mask = getBitMask(targets);
    Index minNonTarg = 0;
    while (getBit(mask, minNonTarg))
        minNonTarg++;
        
    // record which target qubits require swapping
    NatArray newTargs(0);
    for (Nat oldTarg : targets)
        if (oldTarg < psi.logNumAmpsPerNode)
            newTargs.push_back(oldTarg);
        else {
            newTargs.push_back(minNonTarg);
            while (getBit(mask, minNonTarg))
                minNonTarg++;
        }
    
    // perform necessary swaps (each definitely inducing communication)
    for (Nat i=0; i<targets.size(); i++)
        if (newTargs[i] != targets[i])
            distributed_statevector_swapGate(psi, newTargs[i], targets[i]);
            
    // embarrassingly parallel
    local_statevector_manyTargGate(psi, newTargs, gate);
    
    // undo swaps
    for (Nat i=0; i<targets.size(); i++)
        if (newTargs[i] != targets[i])
            distributed_statevector_swapGate(psi, newTargs[i], targets[i]);
}


static void distributed_statevector_pauliTensorOrGadget_subroutine(StateVector &psi, Nat pairRank, Amp powI, Index maskXY, Index maskYZ, Amp thisAmpFac, Amp otherAmpFac) {
    
    comm_exchangeArrays(psi.amps, psi.buffer, pairRank);
    
    Index rankShift = pairRank << psi.logNumAmpsPerNode;
    
    for (Index j0=0; j0<psi.numAmpsPerNode; j0++) {
        Index j1 = j0 ^ maskXY;
        Index i1 = rankShift | j1;
        Nat p1 = getBitMaskParity(i1 & maskYZ);
        Amp b1 = (1. - 2.*p1) * powI;
        psi.amps[j0] = thisAmpFac*psi.amps[j0] + otherAmpFac*b1*psi.buffer[j1];
    }
}


static void distributed_statevector_pauliTensorOrGadget(StateVector &psi, NatArray targets, NatArray paulis, Amp thisAmpFac, Amp otherAmpFac) {
    assert( targets.size() == paulis.size() );
    
    // determine powI = i^(number of Y paulis)
    Amp powI = 1;
    for (Nat pauli : paulis)
        if (pauli == Y)
            powI *= Amp(0,1);
            
    // determine pair rank
    Nat pairRank = psi.rank;
    for (Nat i=0; i<targets.size(); i++)
        if (targets[i] >= psi.logNumAmpsPerNode)
            if (paulis[i] == X || paulis[i] == Y)
                pairRank = flipBit(pairRank, targets[i] - psi.logNumAmpsPerNode);
                
    NatArray suffixTargsXY(0);
    for (Nat i=0; i<targets.size(); i++)
        if (targets[i] < psi.logNumAmpsPerNode)
            if (paulis[i] == X || paulis[i] == Y)
                suffixTargsXY.push_back(targets[i]);
    
    Index maskXY = getBitMask(suffixTargsXY);
    Index maskYZ = 0;
    for (Nat i=0; i<targets.size(); i++)
        if (paulis[i] == Y || paulis[i] == Z)
            maskYZ = flipBit(maskYZ, targets[i]);
                
    if (psi.rank == pairRank)
        local_statevector_pauliTensorOrGadget_subroutine(psi, suffixTargsXY, powI, maskXY, maskYZ, thisAmpFac, otherAmpFac);
    else
        distributed_statevector_pauliTensorOrGadget_subroutine(psi, pairRank, powI, maskXY, maskYZ, thisAmpFac, otherAmpFac);
}


static void distributed_statevector_pauliTensor(StateVector &psi, NatArray targets, NatArray paulis) {
    Amp thisAmpFac = 0.;
    Amp otherAmpFac = 1.;
    
    distributed_statevector_pauliTensorOrGadget(psi, targets, paulis, thisAmpFac, otherAmpFac);
}


static void distributed_statevector_pauliGadget(StateVector &psi, NatArray targets, NatArray paulis, Real theta) {
    Amp thisAmpFac = Amp(cos(theta), 0);
    Amp otherAmpFac = Amp(0, sin(theta));
    
    distributed_statevector_pauliTensorOrGadget(psi, targets, paulis, thisAmpFac, otherAmpFac);
}


static void distributed_statevector_phaseGadget(StateVector &psi, NatArray targets, Real theta) {
    
    local_statevector_phaseGadget(psi, targets, theta);
}



#endif // DISTRIBUTED_STATEVECTOR_HPP