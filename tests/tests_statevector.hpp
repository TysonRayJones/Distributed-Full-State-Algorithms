#ifndef TESTS_STATEVECTOR_HPP
#define TESTS_STATEVECTOR_HPP


#include "types.hpp"
#include "states.hpp"
#include "distributed_statevector.hpp"

#include "test_utilities.hpp"
#include "catch_amalgamated.hpp"


int NUM_QUBITS_PSI = 6;
int NUM_TRIALS_PER_PSI_TEST = 5000;


#define PREPARE_PSI_TEST(psiVar, refVar) \
    GENERATE( range(0,NUM_TRIALS_PER_PSI_TEST) ); \
    StateVector psiVar = StateVector(NUM_QUBITS_PSI); \
    psiVar.setRandomAmps(); \
    AmpArray refVar = psiVar.getAllVecAmps();

    
TEST_CASE( "statevector_oneTargGate" ) {
        
    PREPARE_PSI_TEST( psi, ref );
    
    AmpMatrix gate = getRandomMatrix( powerOf2(1) );
    Nat target = getRandomNat(0, NUM_QUBITS_PSI);

    distributed_statevector_oneTargGate(psi, target, gate);
    applyGateToLocalState(ref, {}, {target}, gate);

    REQUIRE( psi.agreesWith(ref) );
}


TEST_CASE( "statevector_manyCtrlOneTargGate") {
    
    PREPARE_PSI_TEST( psi, ref );
    
    AmpMatrix gate = getRandomMatrix( powerOf2(1) );
    Nat target = getRandomNat(0, NUM_QUBITS_PSI);
    Nat numCtrls = getRandomNat(1, NUM_QUBITS_PSI-1);
    NatArray controls = getRandomUniqueNatArray(0, NUM_QUBITS_PSI, numCtrls, target);

    distributed_statevector_manyCtrlOneTargGate(psi, controls, target, gate);
    applyGateToLocalState(ref, controls, {target}, gate);

    REQUIRE( psi.agreesWith(ref) );
}


TEST_CASE( "statevector_swapGate" ) {

    PREPARE_PSI_TEST( psi, ref );

    AmpMatrix gate = {{1,0,0,0}, {0,0,1,0}, {0,1,0,0}, {0,0,0,1}};
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_PSI, 2);

    distributed_statevector_swapGate(psi, targets[0], targets[1]);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( psi.agreesWith(ref) );
}


TEST_CASE( "statevector_manyTargGate") {
    
    PREPARE_PSI_TEST( psi, ref );

    Nat maxNumTargs = NUM_QUBITS_PSI - psi.logNumNodes;
    Nat numTargs = getRandomNat(1, maxNumTargs + 1);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_PSI, numTargs);
    AmpMatrix gate = getRandomMatrix( powerOf2(numTargs) );

    distributed_statevector_manyTargGate(psi, targets, gate);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( psi.agreesWith(ref) );
}


TEST_CASE( "statevector_pauliTensor" ) {

    PREPARE_PSI_TEST( psi, ref );

    Nat numTargs = getRandomNat(1, NUM_QUBITS_PSI + 1);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_PSI, numTargs);
    NatArray paulis = getRandomNatArray(1, 4, numTargs);
    ensureNotAllPauliZ(paulis);
    AmpMatrix gate = getKroneckerProductOfPaulis(paulis);

    distributed_statevector_pauliTensor(psi, targets, paulis);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( psi.agreesWith(ref) );
}


TEST_CASE( "statevector_pauliGadget" ) {

    PREPARE_PSI_TEST( psi, ref );

    Nat numTargs = getRandomNat(1, NUM_QUBITS_PSI + 1);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_PSI, numTargs);
    NatArray paulis = getRandomNatArray(1, 4, numTargs);
    ensureNotAllPauliZ(paulis);
    Real theta = getRandomReal(-PI, PI);
    AmpMatrix tensor = getKroneckerProductOfPaulis(paulis);
    AmpMatrix gate = getExponentialOfPauliTensor(theta, tensor);

    distributed_statevector_pauliGadget(psi, targets, paulis, theta);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( psi.agreesWith(ref) );
}


TEST_CASE( "statevector_phaseGadget") {
    
    PREPARE_PSI_TEST( psi, ref );

    Nat numTargs = getRandomNat(1, NUM_QUBITS_PSI + 1);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_PSI, numTargs);
    Real theta = getRandomReal(-PI, PI);
    AmpMatrix tensor = getKroneckerProductOfPaulis(NatArray(numTargs, 3));
    AmpMatrix gate = getExponentialOfPauliTensor(theta, tensor);

    distributed_statevector_phaseGadget(psi, targets, theta);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( psi.agreesWith(ref) );
}


#endif // TESTS_STATEVECTOR_HPP