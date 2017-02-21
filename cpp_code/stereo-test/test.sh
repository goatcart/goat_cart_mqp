#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

echo "CMAKE"
cmake ..

echo "MAKE"
make
if [ $? -eq 0 ]; then
    echo "MAKE DONE, RUNNING"
    cp -f ../params.yml .
    ./stereo_vision
fi
