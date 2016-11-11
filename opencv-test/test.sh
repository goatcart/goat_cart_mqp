#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
    cd build
    echo "CMAKE"
    cmake ..
else
    cd build
fi

echo "MAKE"
make
echo "RUNNING"
cp -f ../params.yml .
./stereo_vision
