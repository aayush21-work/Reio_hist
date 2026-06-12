#!/bin/fish
echo building binaries
echo --------------------
cd class_public
make class -j
cd ..

cd music
make
cd ..

cd  script
pip install . --break-system-packages
cd ..

echo binaries built
echo ---------------------------------------------------
