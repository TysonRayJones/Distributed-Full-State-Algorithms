
warns='-Wall -Wextra -Wno-unused-function'
args="-Isrc -Itests -std=c++17 -O3 $warns"

echo "compiling test..."
mpic++ $args tests/tests.cpp -o test $*
