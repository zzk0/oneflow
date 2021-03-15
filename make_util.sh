cd ./build && rm -rf CTestTestfile.cmake cmake_install.cmake  CMakeCache.txt  CMakeFiles  cached_filename_list
cmake .. -DTHIRD_PARTY_MIRROR=aliyun  -DCMAKE_BUILD_TYPE=Debug  -DBUILD_PROFILER=ON  -DCMAKE_EXPORT_COMPILE_COMMANDS=1

