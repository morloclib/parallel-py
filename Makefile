all:
	morloc make -o test test.loc
	./test

clean:
	rm -rf test test-build pools/
