ALL_C_FILES="$(find src -type f -name "*.c") $(find test/ -type f -name "*.c")"
gcc -o out/test $ALL_C_FILES -lm -DUNIT_TESTS

if [ $? -eq 0 ]; then
    out/test
fi
