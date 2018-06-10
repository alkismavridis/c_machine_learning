ALL_C_FILES="$(find src -type f -name "*.c")"
gcc -o ./out/main $ALL_C_FILES -lm
