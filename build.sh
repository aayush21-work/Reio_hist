#!/usr/bin/env bash
set -euo pipefail

ROOT="$PWD"
LINE_MAX=8192

# ---------- fetch ----------
[ -d class_public ] || git clone https://github.com/lesgourg/class_public.git
[ -d script ]       || git clone https://bitbucket.org/rctirthankar/script
[ -d music ]        || git clone https://bitbucket.org/ohahn/music.git

# ---------- CLASS ----------
cd "$ROOT/class_public"


sed -i 's/_LINE_LENGTH_MAX_ 1024/_LINE_LENGTH_MAX_ 8192/'         include/parser.h
sed -i 's/_ARGUMENT_LENGTH_MAX_ 1024/_ARGUMENT_LENGTH_MAX_ 8192/' include/parser.h
grep -E "_LINE_LENGTH_MAX_|_ARGUMENT_LENGTH_MAX_" include/parser.h

make clean && make -j                          # not 'make class' -- need libclass.a too

cd python
pip install . --force-reinstall --no-build-isolation
cd "$ROOT/class_public"

# external data must be copied AFTER the wrapper is installed
SITE=$(python -c "import classy, os; print(os.path.dirname(os.path.dirname(classy.__file__)))")
cp -r external "$SITE/"
echo "external/ -> $SITE"

cd "$ROOT"

# MUSIC + SCRIPT (validation leg only) 
if [ "${WITH_NBODY:-0}" = "1" ]; then
    (cd music && make)
    (cd script && pip install .)
    cp reiotest.conf music/
fi

cp reiotest.ini reiotest_1.ini class_public/

cd "$ROOT/class_public"
make -j
python build_test.py

echo "done."
