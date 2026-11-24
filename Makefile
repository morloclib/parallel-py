.PHONY: all clean bench unit

all: unit
	morloc make -o test test.loc
	./test

clean:
	rm -rf test pools/ bench bench-cross *-build* *.dat *.log __pycache__ unit/__pycache__

bench:
	bash ../parallel/bench/run.sh py

unit:
	python3 unit/test_parallel.py
	if command -v taskset > /dev/null; then MLCPAR_EXPECT_CPUS=2 taskset -c 0-1 python3 unit/test_parallel.py; fi
