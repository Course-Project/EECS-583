# convert source code to bitcode (IR)
clang -emit-llvm -c $1.c -o $1.bc
# instrument profiler
opt -pgo-instr-gen -instrprof $1.bc -o $1.prof.bc
# generate binary executable with profiler embedded
clang -fprofile-instr-generate $1.prof.bc -o $1.prof
# run the proram to collect profile data. After this run, a filed named default.profraw is created, which contains the profile data
./$1.prof $2 > /dev/null
# convert the profile data so that we can use it in LLVM
llvm-profdata merge -output=$1.pgo.profdata default.profraw
# use the profile data as input, and apply the HW1 pass
opt -pgo-instr-use -pgo-test-profile-file=$1.pgo.profdata -load ./HW1/HW1PASS/LLVMHW1PASS.so -hw1 $1.bc > /dev/null

# rename the output
mv output.opcstats "583"$1.opcstats
echo 'See your output: '$1.opcstats

cat "583"$1.opcstats
echo ''

if [ $1 = "compress" ]; then
    cp `dirname $2`/compress.in.bak $2
fi

