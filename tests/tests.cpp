#define CATCH_CONFIG_RUNNER
#define CATCH_AMALGAMATED_CUSTOM_MAIN
#include "catch_amalgamated.hpp"

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