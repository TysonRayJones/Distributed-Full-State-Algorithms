#ifndef TESTS_DENSITYMATRIX_HPP
#define TESTS_DENSITYMATRIX_HPP


#include "types.hpp"
#include "states.hpp"
#include "distributed_densitymatrix.hpp"

#include "test_utilities.hpp"
#include "catch_amalgamated.hpp"

#include <algorithm>


int NUM_QUBITS_RHO = 5;
int NUM_TRIALS_PER_RHO_TEST = 5000;


#define PREPARE_RHO_TEST(rhoVar, refVar) \
    GENERATE( range(0,NUM_TRIALS_PER_RHO_TEST) ); \
    DensityMatrix rhoVar = DensityMatrix(NUM_QUBITS_RHO); \
    rhoVar.setRandomAmps(); \
    AmpMatrix refVar = rhoVar.getAllMatrAmps();


TEST_CASE( "densitymatrix_manyTargGate" ) {
        
    PREPARE_RHO_TEST( rho, ref );
    
    Nat maxNumTargs = NUM_QUBITS_RHO / 2;
    Nat numTargs = getRandomNat(1, maxNumTargs);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, numTargs);
    AmpMatrix gate = getRandomMatrix( powerOf2(numTargs) );

    distributed_densitymatrix_manyTargGate(rho, targets, gate);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_swapGate" ) {

    PREPARE_RHO_TEST( rho, ref );

    AmpMatrix gate = {{1,0,0,0}, {0,0,1,0}, {0,1,0,0}, {0,0,0,1}};
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, 2);

    distributed_densitymatrix_swapGate(rho, targets[0], targets[1]);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_pauliTensor" ) {

    PREPARE_RHO_TEST( rho, ref );

    Nat numTargs = getRandomNat(1, NUM_QUBITS_RHO);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, numTargs);
    NatArray paulis = getRandomNatArray(1, 4, numTargs);
    ensureNotAllPauliZ(paulis);
    AmpMatrix gate = getKroneckerProductOfPaulis(paulis);

    distributed_densitymatrix_pauliTensor(rho, targets, paulis);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_pauliGadget" ) {

    PREPARE_RHO_TEST( rho, ref );

    Nat numTargs = getRandomNat(1, NUM_QUBITS_RHO);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, numTargs);
    NatArray paulis = getRandomNatArray(1, 4, numTargs);
    ensureNotAllPauliZ(paulis);
    Real theta = getRandomReal(-PI, PI);
    AmpMatrix tensor = getKroneckerProductOfPaulis(paulis);
    AmpMatrix gate = getExponentialOfPauliTensor(theta, tensor);

    distributed_densitymatrix_pauliGadget(rho, targets, paulis, theta);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_phaseGadget") {
    
    PREPARE_RHO_TEST( rho, ref );

    Nat maxNumTargs = NUM_QUBITS_RHO / 2;
    Nat numTargs = getRandomNat(1, maxNumTargs);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, numTargs);
    Real theta = getRandomReal(-PI, PI);
    AmpMatrix tensor = getKroneckerProductOfPaulis(NatArray(numTargs, 3));
    AmpMatrix gate = getExponentialOfPauliTensor(theta, tensor);

    distributed_densitymatrix_phaseGadget(rho, targets, theta);
    applyGateToLocalState(ref, {}, targets, gate);

    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_krausMap" ) {

    PREPARE_RHO_TEST( rho, ref );

    Nat maxNumTargs = NUM_QUBITS_RHO / 2;
    Nat numTargs = getRandomNat(1, maxNumTargs);
    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, numTargs);
    Nat numKrausOps = getRandomNat(1, 10);
    MatrixArray krausOps = getRandomMatrices(powerOf2(numTargs), numKrausOps);

    distributed_densitymatrix_krausMap(rho, krausOps, targets);
    applyKrausMapToLocalState(ref, targets, krausOps);

    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_oneQubitDephasing" ) {

    PREPARE_RHO_TEST( rho, ref );

    Nat target = getRandomNat(0, NUM_QUBITS_RHO);
    Real prob = getRandomReal(0, 1/2.);
    MatrixArray krausOps(2);
    krausOps[0] = sqrt(1-prob) * matrI;
    krausOps[1] = sqrt(prob)   * matrZ;

    distributed_densitymatrix_oneQubitDephasing(rho, target, prob);
    applyKrausMapToLocalState(ref, {target}, krausOps);
    
    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_twoQubitDephasing" ) {

    PREPARE_RHO_TEST( rho, ref );

    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, 2);
    Real prob = getRandomReal(0, 3/4.);
    MatrixArray krausOps(4);
    krausOps[0] = sqrt(1-prob)  * getKroneckerProduct(matrI, matrI);
    krausOps[1] = sqrt(prob/3.) * getKroneckerProduct(matrI, matrZ);
    krausOps[2] = sqrt(prob/3.) * getKroneckerProduct(matrZ, matrI);
    krausOps[3] = sqrt(prob/3.) * getKroneckerProduct(matrZ, matrZ);

    distributed_densitymatrix_twoQubitDephasing(rho, targets[0], targets[1], prob);
    applyKrausMapToLocalState(ref, targets, krausOps);
    
    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_oneQubitDepolarising" ) {

    PREPARE_RHO_TEST( rho, ref );

    Nat target = getRandomNat(0, NUM_QUBITS_RHO);
    Real prob = getRandomReal(0, 1/2.);
    MatrixArray krausOps(4);
    krausOps[0] = sqrt(1-prob)  * matrI;
    krausOps[1] = sqrt(prob/3.) * matrX;
    krausOps[2] = sqrt(prob/3.) * matrY;
    krausOps[3] = sqrt(prob/3.) * matrZ;

    distributed_densitymatrix_oneQubitDepolarising(rho, target, prob);
    applyKrausMapToLocalState(ref, {target}, krausOps);
    
    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_twoQubitDepolarising" ) {

    PREPARE_RHO_TEST( rho, ref );

    NatArray targets = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, 2);
    Real prob = getRandomReal(0, 3/4.);
    MatrixArray krausOps(16);
    Nat i=0;
    for (AmpMatrix matr1 : {matrI, matrX, matrY, matrZ})
        for (AmpMatrix matr2 : {matrI, matrX, matrY, matrZ})
            krausOps[i++] = sqrt(prob/15.) * getKroneckerProduct(matr1, matr2);
    krausOps[0] = sqrt(1-16/15.) * getIdentityMatrix( powerOf2(2) );

    distributed_densitymatrix_twoQubitDepolarising(rho, targets[0], targets[1], prob);
    applyKrausMapToLocalState(ref, targets, krausOps);
    
    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_damping" ) {

    PREPARE_RHO_TEST( rho, ref );

    Nat target = getRandomNat(0, NUM_QUBITS_RHO);
    Real prob = getRandomReal(0, 1/2.);
    MatrixArray krausOps(2);
    krausOps[0] = {{1,0},{0,sqrt(1-prob)}};
    krausOps[1] = {{0,sqrt(prob)},{0,0}};

    distributed_densitymatrix_damping(rho, target, prob);
    applyKrausMapToLocalState(ref, {target}, krausOps);
    
    REQUIRE( rho.agreesWith(ref) );
}


TEST_CASE( "densitymatrix_expecPauliString" ) {

    PREPARE_RHO_TEST( rho, ref );

    Nat numQubits = NUM_QUBITS_RHO;
    Nat numTerms = getRandomNat(1, 30);
    Nat numPaulis = numTerms * numQubits;
    NatArray paulis = getRandomNatArray(0, 4, numPaulis);
    RealArray coeffs = getRandomRealArray(-10, 10, numTerms);

    Amp expec1 = distributed_densitymatrix_expecPauliString(rho, coeffs, paulis);

    AmpMatrix pauliStringMatr = getZeroMatrix( powerOf2(numQubits) );
    for (Nat i=0; i<numTerms; i++) {
        NatArray prodPaulis = NatArray(paulis.begin() + i*numQubits, paulis.begin() + (i+1)*numQubits);
        AmpMatrix prodMatrix = getKroneckerProductOfPaulis(prodPaulis);
        pauliStringMatr = pauliStringMatr +  coeffs[i] * prodMatrix;
    }
    ref = pauliStringMatr * ref;

    Amp expec2 = 0;
    for (Nat r=0; r<ref.size(); r++)
        expec2 += ref[r][r];

    REQUIRE_THAT( real(expec1), Catch::Matchers::WithinAbs(real(expec2), 1e-12) );
    REQUIRE_THAT( imag(expec1), Catch::Matchers::WithinAbs(imag(expec2), 1e-12) );
}


TEST_CASE( "densitymatrix_partialTrace" ) {

    PREPARE_RHO_TEST( rho, ref );

    // cannot trace out more qubits than can fit in the suffix sub-register
    Nat maxNumTargs = NUM_QUBITS_RHO - rho.logNumNodes;
    
    Nat numTargs = getRandomNat(1, maxNumTargs+1);
    NatArray targs = getRandomUniqueNatArray(0, NUM_QUBITS_RHO, numTargs);
    std::sort(targs.begin(), targs.end());

    DensityMatrix rhoOut = distributed_densitymatrix_partialTrace(rho, targs);

    // serially populate the reference reduced matrix
    AmpMatrix refOut = getZeroMatrix( powerOf2(NUM_QUBITS_RHO - numTargs) );
    
    for (Index i=0; i<refOut.size(); i++) {
        for (Index j=0; j<refOut.size(); j++) {
            Index i0 = insertBits(i, targs, 0);
            Index j0 = insertBits(j, targs, 0);

            refOut[i][j] = 0;
            for (Index k=0; k<powerOf2(numTargs); k++) {
                Index ik = setBits(i0, targs, k);
                Index jk = setBits(j0, targs, k);
                refOut[i][j] += ref[ik][jk];
            }
        }
    }

    REQUIRE( rhoOut.agreesWith(refOut) );
}


#endif // TESTS_DENSITYMATRIX_HPP