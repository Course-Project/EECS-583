# convert source code to bitcode (IR)
clang -emit-llvm -c $1.c -o $1.bc
# canonicalize natural loops
opt -loop-simplify $1.bc -o $1.ls.bc
# instrument profiler
opt -pgo-instr-gen -instrprof $1.ls.bc -o $1.ls.prof.bc
# generate binary executable with profiler embedded
clang -fprofile-instr-generate $1.ls.prof.bc -o $1.prof
# run the proram to collect profile data. After this run, a filed named default.profraw is created, which contains the profile data
# this run will also generate the correct output
./$1.prof > $1_correct_output
# convert the profile data so that we can use it in LLVM
llvm-profdata merge -output=$1.pgo.profdata default.profraw
# use the profile data as input, and apply the HW2 pass
opt -pgo-instr-use -pgo-test-profile-file=$1.pgo.profdata -load HW2/HW2PASS/LLVMHW2FPLICM.so -fplicm $1.ls.bc > $1.final.bc
# generate binary executable after FPLICM
clang $1.final.bc -o $1.final
# see the new output
./$1.final > $1_pass_output
# verify your final output
diff $1_pass_output $1_correct_output
