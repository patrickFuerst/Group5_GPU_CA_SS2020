#!/bin/sh

echo "Clean build directory"

rm -rf ./build

echo "Build all applications!"

mkdir build 
cmake -B "./build" -DCMAKE_BUILD_TYPE=Release
cmake --build  build  --config Release -j 8

echo "Install all applications" 
cmake --install build --config Release 



