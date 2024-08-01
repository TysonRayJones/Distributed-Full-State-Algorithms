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

using namespace std;
using namespace std::chrono;


/*
 * Compares 
 * - oneTargetGate,
 * - manyCtrlOneTargGate
 * - manyTargGate
 * when given no controls and a single target.
 */


const int NUM_REPS = 100;



int main(int argc, char* argv[]) {

    comm_init();

    Nat numQubits = atoi(argv[1]);
    StateVector state = StateVector(numQubits);
    AmpMatrix matrix;

    for (Index i=0; i<state.numAmpsPerNode; i++)
        state.amps[i] = 1;



    // try to eliminate warm-up effects
    for (Nat n=0; n<NUM_REPS; n++) {
        for (Nat t=0; t<numQubits; t++) {
            matrix = getRandomMatrix( powerOf2(1) );
            distributed_statevector_oneTargGate(state, t, matrix);
            distributed_statevector_manyCtrlOneTargGate(state, {}, t, matrix);
            distributed_statevector_manyTargGate(state, {t}, matrix);
        }
    }
    


    comm_synch();
    auto start = high_resolution_clock::now();

    for (Nat n=0; n<NUM_REPS; n++) {
        for (Nat t=0; t<numQubits; t++) {
            matrix = getRandomMatrix( powerOf2(1) );
            distributed_statevector_oneTargGate(state, t, matrix);
        }
    }

    comm_synch();
    auto stop = high_resolution_clock::now();
    auto durA = duration_cast<microseconds>(stop - start).count();



    comm_synch();
    start = high_resolution_clock::now();

    for (Nat n=0; n<NUM_REPS; n++) {
        for (Nat t=0; t<numQubits; t++) {
            matrix = getRandomMatrix( powerOf2(1) );
            distributed_statevector_manyCtrlOneTargGate(state, {}, t, matrix);
        }
    }

    comm_synch();
    stop = high_resolution_clock::now();
    auto durB = duration_cast<microseconds>(stop - start).count();
    


    comm_synch();
    start = high_resolution_clock::now();

    for (Nat n=0; n<NUM_REPS; n++) {
        for (Nat t=0; t<numQubits; t++) {
            distributed_statevector_manyTargGate(state, {t}, matrix);
        }
    }

    comm_synch();
    stop = high_resolution_clock::now();
    auto durC = duration_cast<microseconds>(stop - start).count();


    if (state.rank == 0) {
        cout << "durA (oneTargGate):         " << durA << endl;
        cout << "durB (manyCtrlOneTargGate): " << durB << endl;
        cout << "durC (manyTargGate):        " << durC << endl;
    }


    comm_synch();
    comm_end();
    return 0;
}