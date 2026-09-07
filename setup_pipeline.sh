#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "getting the repos...."

echo "getting class"
[ -d "$ROOT/class_public" ] || git clone https://github.com/lesgourg/class_public.git
echo "done."

echo "getting script"
[ -d "$ROOT/script" ] || git clone https://bitbucket.org/rctirthankar/script
echo "done."

echo "getting music"
[ -d "$ROOT/music" ] || git clone https://bitbucket.org/ohahn/music.git
echo "done."

echo "building binaries"
echo "--------------------"

cd "$ROOT/class_public"
make -j
cd "$ROOT"

cd "$ROOT/music"
make
cd "$ROOT"

cd "$ROOT/script"
pip install . --break-system-packages
cd "$ROOT"

echo "binaries built"
echo "---------------------------------------------------"

echo "relocating ini files"

mv "$ROOT/reiotest.ini" "$ROOT/reiotest_1.ini" "$ROOT/class_public"
mv "$ROOT/reiotest.conf" "$ROOT/music"

echo "Done."
