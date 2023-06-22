
#include "types.hpp"
#include "states.hpp"
#include "communication.hpp"
#include "test_utilities.hpp"
#include "catch.hpp"

#include "distributed_densitymatrix.hpp"


int NUM_QUBITS_RHO = 5;


#define PREPARE_RHO_TEST(rhoVar, refVar) \
    DensityMatrix rhoVar = DensityMatrix(NUM_QUBITS_RHO); \
    rhoVar.setRandomAmps(); \
    AmpMatrix refVar = rhoVar.getAllMatrAmps();








