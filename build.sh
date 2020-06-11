
echo "Clean build directory"

rm -rf ./build

echo "Build all applications!"

mkdir build 
cmake -B "./build" 
cmake --build  build --target ALL_BUILD --config Release


echo "Install all applications" 
cmake --install build 



