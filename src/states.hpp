
#ifndef STATES_HPP
#define STATES_HPP

#include "types.hpp"
#include "bit_maths.hpp"
#include "misc.hpp"
#include "communication.hpp"

#include <assert.h>



class StateVector {
    
public:
    Nat rank;
    Nat numNodes;
    Nat logNumNodes;
    
    Nat numQubits;
    Index numAmpsPerNode;
    Index logNumAmpsPerNode;
    
    AmpArray amps;
    AmpArray buffer;
    
    AmpArray getAllVecAmps();
    void setRandomAmps();
    void printAmps();
    bool agreesWith(AmpArray allAmps, Real tol);
    
    StateVector(Nat numQubits) {
        
        // enforces >=1 statevec amp per node, and >=1 density-matrix column per node
        assert( powerOf2(numQubits) >= comm_getNumNodes() );
        
        this->rank = comm_getRank();
        this->numNodes = comm_getNumNodes();
        
        this->numQubits = numQubits;
        this->logNumNodes = logBase2(this->numNodes);
        
        this->logNumAmpsPerNode = (numQubits - this->logNumNodes);
        this->numAmpsPerNode = powerOf2(this->logNumAmpsPerNode);
        
        this->amps = AmpArray(this->numAmpsPerNode, 0);
        this->buffer = AmpArray(this->numAmpsPerNode, 0);
    }
};



class DensityMatrix : public StateVector {

public:
    
    AmpMatrix getAllMatrAmps();
    bool agreesWith(AmpMatrix allAmps, Real tol);
    
    DensityMatrix(Nat numQubits) : StateVector(numQubits) {
        
        // increase the number of initialised statevector amps 
        this->logNumAmpsPerNode = (2*numQubits - this->logNumNodes);
        this->numAmpsPerNode = powerOf2(this->logNumAmpsPerNode);
        
        this->amps.resize(this->numAmpsPerNode, 0);
        this->buffer.resize(this->numAmpsPerNode, 0);
    }
};



#endif // STATES_HPP