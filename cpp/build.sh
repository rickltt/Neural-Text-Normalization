#/bin/bash

# build
rm -rf build
mkdir -p build
cd build
cmake ..
make -j8
cd ..

# run
# echo "---Running example----"
# ./build/example