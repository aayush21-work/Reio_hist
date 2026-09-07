#!/usr/bin/env bash
set -euo pipefail

ROOT="$PWD"
LINE_MAX=8192


[ -d class_public ] || git clone https://github.com/lesgourg/class_public.git
[ -d script ]       || git clone https://bitbucket.org/rctirthankar/script
[ -d music ]        || git clone https://bitbucket.org/ohahn/music.git

echo Building Class.....

cd "$ROOT/class_public"


sed -i 's/_LINE_LENGTH_MAX_ 1024/_LINE_LENGTH_MAX_ 8192/'         include/parser.h
sed -i 's/_ARGUMENT_LENGTH_MAX_ 1024/_ARGUMENT_LENGTH_MAX_ 8192/' include/parser.h
grep -E "_LINE_LENGTH_MAX_|_ARGUMENT_LENGTH_MAX_" include/parser.h

make clean && make -j                     

cd python
pip install . --force-reinstall --no-build-isolation
cd "$ROOT/class_public"


SITE=$(python -c "import classy, os; print(os.path.dirname(os.path.dirname(classy.__file__)))")
cp -r external "$SITE/"
echo "external/ -> $SITE"

cd "$ROOT/music"
make
cd "$ROOT"

cd "$ROOT/script"
pip install . --break-system-packages
cd "$ROOT"

echo "binaries built"
echo "---------------------------------------------------"

echo "relocating ini files"

cp "$ROOT/reiotest.ini" "$ROOT/reiotest_1.ini" "$ROOT/class_public"
cp "$ROOT/reiotest.conf" "$ROOT/music"

echo "Done."

