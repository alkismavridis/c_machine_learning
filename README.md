# c_machine_learning

1. What is this?
A neural network library (plus a basic CLI inteface to interact with it) written in C.
I wrote this for fun, and is not intended to be a preffessional machine learning tool.
Nevertheless, I have written unit tests, and it looks like working fine, so I thought that meybe it is usefull for someone, and made it public.

If you want to include this as a library in your project, and ignore the CLI, just delete the src/cli directory and the Main.c, and you are done.
Its now a library.

PS: This project has no dependencies to other libraries.



2. Compile:
I compiled it using gcc. I am not sure if clang would work.

In any case, to compile this project, first download it, open up a terminal in the root directory, and run
./scripts/buildAndRun.sh

If you download this, you may have to give permissions to the script files first. So you may run before compiling:
chmod +x scripts/*
