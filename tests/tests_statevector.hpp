
#include "types.hpp"
#include "states.hpp"
#include "communication.hpp"
#include "test_utilities.hpp"
#include "catch.hpp"

#include "distributed_statevector.hpp"


int NUM_QUBITS_PSI = 5;


#define PREPARE_PSI_TEST(psiVar, refVar) \
    StateVector psiVar = StateVector(NUM_QUBITS_PSI); \
    psiVar.setRandomAmps(); \
    AmpArray refVar = psiVar.getAllVecAmps();

    
TEST_CASE( "statevector_oneTargGate" ) {
        
    PREPARE_PSI_TEST(psi, ref);
    
    AmpMatrix gate = getRandomMatrix( powerOf2(1) );
    Nat target = GENERATE_COPY( range(0, NUM_QUBITS_PSI) );

    distributed_statevector_oneTargGate(psi, target, gate);
    applyGateToLocalStateVec(ref, {}, {target}, gate);

    REQUIRE( psi.agreesWith(ref) );
}


TEST_CASE( "statevector_manyCtrlOneTargGate") {
    
    PREPARE_PSI_TEST(psi, ref);
    
    AmpMatrix gate = getRandomMatrix( powerOf2(1) );
    Nat target = GENERATE_COPY( range(0, NUM_QUBITS_PSI) );
    
    GENERATE( range(0,10) );
    NatArray controls = getRandom
    
    distributed_statevector_manyCtrlOneTargGate(psi, controls, target, gate);


}
