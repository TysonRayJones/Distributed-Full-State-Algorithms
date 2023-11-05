
Distributed Simulation of Statevectors and Density Matrices
===========================================================

> Tyson Jones, Balint Koczor, Simon C. Benjamin  
> Department of Materials, University of Oxford  
> Quantum Motion Technologies Ltd


This repository contains `C++` implementations of the multithreaded, distributed algorithms presented in [this manuscript](TODO), and unit tests using [Catch2](https://github.com/catchorg/Catch2). If this code is useful to you, please cite
```
TODO arxiv bibtex
```

# Types

The below API makes use of the following custom types defined in [`types.hpp`](src/types.hpp), wherein you can vary their precision.

| Type   | Use           | Default
|--------|---------------|----------|
| `Real` | A real scalar | `double` |
| `Nat`  | A natural scalar | `unsigned int` |
| `Index` | A state index | `long long unsigned int` |
| `Amp` | A complex scalar | `std::complex<Real>` |

We also define arrays and matrices of these types, such as `NatArray`, which are merely eye-candy for `std::vector<Nat>`.


# API

Before calling any of the below functions, you should initialise MPI with `comm_init()`, and before exiting, finalise with `comm_end()`.

Instantiate a quantum state via:
```C++
StateVector psi = StateVector(numQubits);
DensityMatrix rho = DensityMatrix(numQubits);
```

Statevectors can be passed to the below unitary functions, prefixed with `distributed_statevector_`.

- ```C++
  oneTargGate(StateVector psi, Nat target, AmpMatrix gate)
  ```
- ```C++
  manyCtrlOneTargGate(StateVector psi, NatArray controls, Nat target, AmpMatrix gate)
  ```
- ```C++
  swapGate(StateVector psi, Nat qb1, Nat qb2)
  ```
- ```C++
  manyTargGate(StateVector psi, NatArray targets, AmpMatrix gate)
  ```
- ```C++
  pauliTensor(StateVector psi, NatArray targets, NatArray paulis)
  ```
- ```C++
  pauliGadget(StateVector psi, NatArray targets, NatArray paulis, Real theta)
  ```
- ```C++
  phaseGadget(StateVector psi, NatArray targets, Real theta)
  ```

Density matrices can be passed to the below functions, prefixed with `distributed_densitymatrix_`.

- ```C++
  manyTargGate(DensityMatrix rho, NatArray targets, AmpMatrix gate)
  ```
- ```C++
  swapGate(DensityMatrix rho, Nat qb1, Nat qb2)
  ```
- ```C++
  pauliTensor(DensityMatrix rho, NatArray targets, NatArray paulis)
  ```
- ```C++
  pauliGadget(DensityMatrix rho, NatArray targets, NatArray paulis, Real theta)
  ```
- ```C++
  phaseGadget(DensityMatrix rho, NatArray targets, Real theta)
  ```
- ```C++
  phaseGadget(DensityMatrix rho, NatArray targets, Real theta)
  ```
- ```C++
  oneQubitDephasing(DensityMatrix rho, Nat qb, Real prob)
  ```
- ```C++
  twoQubitDephasing(DensityMatrix rho, Nat qb1, Nat qb2, Real prob)
  ```
- ```C++
  oneQubitDepolarising(DensityMatrix rho, Nat qb, Real prob)
  ```
- ```C++
  twoQubitDepolarising(DensityMatrix rho, Nat qb1, Nat qb2, Real prob)
  ```
- ```C++
  damping(DensityMatrix rho, Nat qb, Real prob)
  ```
- ```C++
  expecPauliString(DensityMatrix rho, RealArray coeffs, NatArray allPaulis)
  ```
- ```C++
  partialTrace(DensityMatrix inRho, NatArray targets)
  ```

View the definition of these functions in the [`src`](/src/) folder.

See an example in [`main.cpp`](main.cpp).


# Compiling

To compile both [`main.cpp`](main.cpp) and the [unit tests](/tests/), simply call
```bash
source ./compile
```
Additionally, set the number of threads (per node) via
```bash
export OMP_NUM_THREADS=24
```
and launch the executables between (e.g.) `16` nodes via
```bash
mpirun -np 16 ./main
```
```bash
mpirun -np 16 ./test
```
You must use a power-of-2 number of nodes.


# License

This repository is licensed under the terms of the MIT license.