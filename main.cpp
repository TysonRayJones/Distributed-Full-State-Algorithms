#include "types.hpp"
#include "states.hpp"
#include "local_statevector.hpp"
#include "distributed_statevector.hpp"
#include "distributed_densitymatrix.hpp"
#include "test_utilities.hpp"

#include <stdio.h>
#include <iostream>
#include <complex>
#include <chrono>

using namespace std::chrono;
using namespace std::complex_literals;



int main() {
    
    comm_init();

    Nat numQubits = 26;
    StateVector state = StateVector(numQubits);

    NatArray targets = {0,6,4,2};
    AmpMatrix matrix = getRandomMatrix( powerOf2(targets.size()) ); 

    auto start = high_resolution_clock::now();
    comm_synch();

    distributed_statevector_manyTargGate(state, targets, matrix);

    comm_synch();
    auto stop = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(stop - start).count();

    rootNodePrint("done in " + std::to_string(dur) + " microseconds\n");

    comm_end();
    return 0;
}