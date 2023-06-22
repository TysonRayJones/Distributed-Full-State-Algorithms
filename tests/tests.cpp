
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "test_utilities.hpp"
#include "tests_statevector.hpp"
#include "tests_densitymatrix.hpp"

#include "communication.hpp"


int main( int argc, char* argv[] ) {
  comm_init();  
  int result = Catch::Session().run( argc, argv ); 
  comm_end();
  return result;
}




/*
std::vector<StateVector> stateVectors;



Index getTotalSystemMemory() {
    
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}


void prepareStateVectors() {
    
    // max register size when copied fully on a node, and all smaller registers also exist
    Nat maxNumQubits = (Nat) log2(getTotalSystemMemory() / sizeof(Amp)) - 1;
    
    printf("maxNumQubits: %u\n", maxNumQubits);
    
    for (Nat numQubits=2; numQubits<=maxNumQubits; numQubits++)
        stateVectors.push_back(StateVector(numQubits));
}
*/


// deploy this to AWS??


// if you make ALL permanent statevetors at once, it only doubles the total memory requirement!
