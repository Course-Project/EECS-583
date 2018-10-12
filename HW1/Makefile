DIRS = HW1
CLEANDIRS = $(DIRS:%=clean-%)
SRC = HW1/HW1PASS/HW1PASS.cpp
SUB_SRC_DIR = "./submission_hw1/src"
SUB_RESULT_DIR = "./submission_hw1/results"

build:
	$(MAKE) -C HW1

.PHONY: SUBDIRS $(CLEANDIRS)

clean: $(CLEANDIRS)
	@echo "Deleting intermediate files ..."
	rm -f *.bc
	rm -f *.prof.bc
	rm -f *.prof
	rm -f default.profraw
	rm -f *.pgo.profdata
	rm -f 583compress/input1/*.Z
	
	@echo "Deleting Vim bak files ..."
	rm -f *\~
	rm -f .*\~

$(CLEANDIRS):
	$(MAKE) -C $(@:clean-%=%) clean

clean-output:
	@echo "Deleting output files ..."
	rm -f *.opcstats

clean-submission:
	@echo "Deleting submission directory ..."
	rm -rf submission_hw1

clean-all: clean clean-output clean-submission

pack: run
	@echo "Packing homework submission files ..."
	@mkdir -p $(SUB_SRC_DIR)
	@mkdir -p $(SUB_RESULT_DIR)
	@cp HW1/HW1PASS/HW1PASS.cpp $(SUB_SRC_DIR)
	@cp 583wc.opcstats 583simple.opcstats 583compress.opcstats $(SUB_RESULT_DIR)
	
	@echo "src:"
	@ls -alh submission_hw1/src
	@echo ""
	@echo "results:"
	@ls -alh submission_hw1/results

run: clean-all build
	@bash ./run.sh wc 583wc/input1/cccp.c 2> /dev/null
	@bash ./run.sh compress 583compress/input1/compress.in 2> /dev/null
	@bash ./run.sh simple 2> /dev/null


