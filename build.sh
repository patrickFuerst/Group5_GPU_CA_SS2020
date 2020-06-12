
echo "Clean build directory"

rm -rf ./build

echo "Build all applications!"


mkdir build 
cmake -B "./build" 

D
D
cmake --build  build  --config Release


echo "Install all applications" 
cmake --install build --config Release 



