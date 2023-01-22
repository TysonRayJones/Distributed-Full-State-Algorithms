
#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include "types.hpp"
#include "bit_maths.hpp"

#ifdef DISTRIBUTED
#include <mpi.h>
#endif



class StateVector {
    
public:
    Nat rank;
    Nat numNodes;
    Nat logNumNodes;
    
    Nat numQubits;
    Index numAmpsPerNode;
    Index logNumAmpsPerNode;
    
    Array amps;
    Array buffer;
    
    StateVector(Nat numQubits) {
        
        #ifdef DISTRIBUTED
        MPI_Comm_rank(MPI_COMM_WORLD, &(this->rank));
        MPI_Comm_size(MPI_COMM_WORLD, &(this->numNodes));
        #else
        this->rank = 0;
        this->numNodes = 1;
        #endif
        
        this->logNumNodes = logBase2(this->numNodes);
        
        this->numQubits = numQubits;
        this->logNumAmpsPerNode = (numQubits - this->logNumNodes);
        this->numAmpsPerNode = powerOf2(this->logNumAmpsPerNode);
        
        this->amps = Array(this->numAmpsPerNode, 0);
        this->buffer = Array(this->numAmpsPerNode, 0);
    }
};



#endif // STRUCTURES_HPP