
#ifndef TEST_UTILITIES_HPP
#define TEST_UTILITIES_HPP

#include "types.hpp"
#include "bit_maths.hpp"
#include "states.hpp"
#include "misc.hpp"

#include <random>
#include <iostream>
#include <algorithm>
#include <assert.h>

// TODO: needed for a single call to allGather() not yet abstracted
#include <mpi.h>



static void printMatrix(AmpMatrix matrix) {
    for (Nat r=0; r<matrix.size(); r++) {
        for (Nat c=0; c<matrix.size(); c++)
            std::cout << matrix[r][c] << " ";
        std::cout << "\n";
    }
}


template<typename T> 
static void printArray(std::vector<T> array) {
    std::cout << "{";
    for (Nat i=0; i<array.size(); i++)
        std::cout << array[i] << ((i<array.size()-1)? " ":"");
    std::cout << "}\n" << std::flush;
}


static void rootNodePrint(std::string msg) {
    comm_synch();
    std::cout << std::flush;
    comm_synch();
    if (comm_getRank() == 0)
        std::cout << msg << "\n" << std::flush;
    comm_synch();
}


template<typename T>
static void allNodesPrintLocalArray(std::vector<T> array) {
    Nat thisRank = comm_getRank();
    Nat numNodes = comm_getNumNodes();
    for (Nat rank=0; rank<numNodes; rank++) {
        if (rank == thisRank) {
            std::cout << "  rank: " << rank << "\n    ";
            printArray(array);
        }
        comm_synch();
    }
    comm_synch();
    std::cout << std::flush;
    comm_synch();
}


static Amp getRandomAmp() {
    
    // generate 2 normally-distributed random numbers via Box-Muller
    Real a = rand()/(Real) RAND_MAX;
    Real b = rand()/(Real) RAND_MAX;
    Real r1 = sqrt(-2 * log(a)) * cos(2 * 3.14159265 * b);
    Real r2 = sqrt(-2 * log(a)) * sin(2 * 3.14159265 * b);
    return Amp(r1, r2);
}


static AmpArray getRandomArray(Index dim) {
    assert( dim > 0 );
    
    AmpArray arr = AmpArray(dim);
    for (Index i=0; i<dim; i++)
        arr[i] = getRandomAmp();
    return arr;
}


static AmpMatrix getRandomMatrix(Index dim) {
    assert( dim > 0 );
    
    AmpMatrix matr = getZeroMatrix(dim);
    for (Index i=0; i<dim; i++)
        for (Index j=0; j<dim; j++)
            matr[i][j] = getRandomAmp();
    return matr;
}


static void setSubMatrix(AmpMatrix &dest, AmpMatrix sub, Index r, Index c) {
    assert( r + sub.size() <= dest.size() );
    assert( c + sub.size() <= dest.size() );
    
    for (Nat i=0; i<sub.size(); i++)
        for (Nat j=0; j<sub.size(); j++)
            dest[r+i][c+j] = sub[i][j];
}


static AmpMatrix getFullSwapMatrix(Nat qb1, Nat qb2, Nat numQb) {
    assert( qb1 < numQb && qb2 < numQb );
    
    if (qb1 > qb2)
        std::swap(qb1, qb2);
        
    if (qb1 == qb2)
        return getIdentityMatrix(powerOf2(numQb));

    AmpMatrix swap;
    
    if (qb2 == qb1 + 1) {
        // qubits are adjacent
        swap = AmpMatrix{{1,0,0,0},{0,0,1,0},{0,1,0,0},{0,0,0,1}};
        
    } else {
        // qubits are distant
        Index block = powerOf2(qb2 - qb1);
        swap = getZeroMatrix(block*2);
        AmpMatrix iden = getIdentityMatrix(block/2);
        
        // Lemma 3.1 of arxiv.org/pdf/1711.09765.pdf
        AmpMatrix p0{{1,0},{0,0}};
        AmpMatrix l0{{0,1},{0,0}};
        AmpMatrix l1{{0,0},{1,0}};
        AmpMatrix p1{{0,0},{0,1}};
        
        /* notating a^(n+1) = identity(1<<n) (otimes) a, we construct the matrix
         * [ p0^(N) l1^N ]
         * [ l0^(N) p1^N ]
         * where N = qb2 - qb1 */
        setSubMatrix(swap, iden % p0, 0, 0);
        setSubMatrix(swap, iden % l0, block, 0);
        setSubMatrix(swap, iden % l1, 0, block);
        setSubMatrix(swap, iden % p1, block, block);
    }
    
    // pad swap with outer identities
    if (qb1 > 0)
        swap = swap % getIdentityMatrix(powerOf2(qb1));
    if (qb2 < numQb-1)
        swap = getIdentityMatrix(powerOf2(numQb-qb2-1)) % swap;
    
    return swap;
}


static bool containsDuplicate(NatArray list1, NatArray list2) {
    NatArray combined = list1;
    combined.insert(combined.end(), list2.begin(), list2.end());
    std::sort(combined.begin(), combined.end());
    for (Nat i=0; i<combined.size()-1; i++)
        if (combined[i+1] == combined[i])
            return true;
    return false;
}


static AmpMatrix getFullGateMatrix(AmpMatrix gateMatr, NatArray ctrls, NatArray targs, Nat numQb) {
    assert( targs.size() > 0 );
    assert( ctrls.size() + targs.size() <= numQb );
    assert( gateMatr.size() == powerOf2(targs.size()) );
    assert( !containsDuplicate(ctrls, targs) );
    assert( *std::max_element(targs.begin(), targs.end()) < numQb );
    if (ctrls.size() > 0)
        assert( *std::max_element(ctrls.begin(), ctrls.end()) < numQb );

    // concatenate all qubits
    NatArray qubits = targs;
    qubits.insert( qubits.end(), ctrls.begin(), ctrls.end() );
    
    // create sub-matrix with qubits #(ctrls+targs), with left-most controlled and 
    // rightmost targetted as if qubits = {0,1,2,3,...}
    Index subDim = powerOf2(qubits.size());
    Index subInd = subDim - gateMatr.size();
    AmpMatrix subMatr = getIdentityMatrix(subDim);
    setSubMatrix(subMatr, gateMatr, subInd, subInd);
    
    // pad sub-matrix to #numQb
    AmpMatrix fullMatr;
    if (numQb == qubits.size())
        fullMatr = subMatr;
    else {
        Index idenDim = powerOf2(numQb - qubits.size());
        fullMatr = getIdentityMatrix(idenDim) % subMatr;
    }
    
    // create swap matrices which sort qubits
    Index fullDim = powerOf2(numQb);
    AmpMatrix swapsMatr = getIdentityMatrix(fullDim);
    AmpMatrix unswapsMatr = getIdentityMatrix(fullDim);
    
    for (Nat q=0; q<qubits.size(); q++) {
        if (qubits[q] != q) {
            AmpMatrix m = getFullSwapMatrix(q, qubits[q], numQb);
            swapsMatr = m * swapsMatr;
            unswapsMatr = unswapsMatr * m;
            
            for (Nat i=q+1; i<qubits.size(); i++)
                if (qubits[i] == q)
                    qubits[i] = qubits[q];
        }
    }
    
    // apply the swaps to our final matrix
    fullMatr = unswapsMatr * fullMatr * swapsMatr;
    return fullMatr;
}


static void applyGateToLocalStateVec(AmpArray& state, NatArray ctrls, NatArray targs, AmpMatrix gateMatr) {
    Nat numQb = logBase2(state.size());
    AmpMatrix fullGateMatr = getFullGateMatrix(gateMatr, ctrls, targs, numQb);
    state = fullGateMatr * state;
}


static void applyGateToLocalDensMatr(AmpMatrix& state, NatArray ctrls, NatArray targs, AmpMatrix gateMatr) {
    Nat numQb = logBase2(state.size());
    AmpMatrix fullGateMatr = getFullGateMatrix(gateMatr, ctrls, targs, numQb);
    state = fullGateMatr * state * getDaggerMatrix(fullGateMatr);
}


AmpArray StateVector::getAllVecAmps() {
    
    comm_synch();
    
    if (this->numNodes == 1)
        return this->amps;
        
    Index numAllAmps = this->numNodes * this->numAmpsPerNode;
    AmpArray allAmps(numAllAmps);

    MPI_Allgather(
        (this->amps).data(), this->numAmpsPerNode, MPI_AMP,
        allAmps.data(),      this->numAmpsPerNode, MPI_AMP,
        MPI_COMM_WORLD);
        
    return allAmps;
}


void StateVector::setRandomAmps() {
    
    comm_synch();
    
    // all nodes generate all amps to keep RNG in-synch
    for (Nat rank=0; rank<this->numNodes; rank++) {
        AmpArray amps = getRandomArray(this->numAmpsPerNode);
        if (this->rank == rank)
            this->amps = amps;
    }
}


void StateVector::printAmps() {
    
    comm_synch();
    
    // nominated rank prints while others wait
    for (Nat rank=0; rank<this->numNodes; rank++) {
        if (this->rank == rank) {
            printf("rank %u:\n", rank);
            printArray(this->amps);
        }
        
        comm_synch();
    }
    
    std::cout << std::flush;
    comm_synch();
}


bool StateVector::agreesWith(AmpArray ref, Real tol=1E-5) {
    assert( ref.size() == powerOf2(this->numQubits) );
    
    comm_synch();
    
    AmpArray allAmps = this->getAllVecAmps();
    
    for (Index i=0; i<ref.size(); i++) {
        Amp dif = ref[i] - allAmps[i];
        if (real(dif) > tol || imag(dif) > tol)
            return false;
    }
    
    return true;
}


AmpMatrix DensityMatrix::getAllMatrAmps() {
    
    comm_synch();
    
    Index dim = powerOf2(this->numQubits);
    AmpArray vec = this->getAllVecAmps();
    AmpMatrix matr = getZeroMatrix(dim);
    
    for (Index r=0; r<dim; r++)
        for (Index c=0; c<dim; c++)
            matr[r][c] = vec[dim*c+r];

    return matr;
}


bool DensityMatrix::agreesWith(AmpMatrix ref, Real tol=1E-5) {
    assert( ref.size() == powerOf2(this->numQubits) );
    
    comm_synch();
    
    AmpMatrix allAmps = this->getAllMatrAmps();
    
    for (Index r=0; r<ref.size(); r++)
        for (Index c=0; c<ref.size(); c++) {
            Amp dif = ref[r][c] - allAmps[r][c];
            if (real(dif) > tol || imag(dif) > tol)
                return false;
        }
    
    return true;
}



#endif // TEST_UTILITIES_HPP
