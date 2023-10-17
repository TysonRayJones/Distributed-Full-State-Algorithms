
conf="-std=c++17 -O3 -fopenmp -march=native"
dirs="-Isrc -Itests -Icatch"

echo "compiling main..."
mpic++ $conf $dirs main.cpp -o main $*

echo "compiling tests..."
mpic++ $conf $dirs tests/tests.cpp catch/catch_amalgamated.cpp -o test $*

export OMP_WAIT_POLICY=active
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=24            # set to #cores per node