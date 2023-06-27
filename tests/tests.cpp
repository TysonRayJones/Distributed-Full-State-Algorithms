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