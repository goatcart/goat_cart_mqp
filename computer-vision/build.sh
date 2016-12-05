#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

echo "CMAKE"
cmake ..

echo "MAKE"
make
