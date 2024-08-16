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

    Nat numQubits = 20;
    Nat numReps = 20;

    StateVector psi = StateVector(numQubits);

    auto start = high_resolution_clock::now();
    comm_synch();

    // XX + YY + ZZ
    for (Nat r=0; r<numReps; r++)
        for (Nat t=0; t<numQubits; t++)
            for (Nat p=1; p<3; p++)
                distributed_statevector_pauliGadget(psi, {t, (t+1)%numQubits}, {p,p}, 0.1*(r+t+p));

    comm_synch();
    auto stop = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(stop - start).count();

    rootNodePrint("done in " + std::to_string(dur) + " microseconds\n");

    comm_end();
    return 0;
}