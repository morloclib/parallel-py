.PHONY: all clean bench

all:
	morloc make -o test test.loc
	./test

clean:
	rm -rf test pools/ bench bench-cross *-build* *.dat *.log __pycache__

bench:
	bash ../parallel/bench/run.sh py
