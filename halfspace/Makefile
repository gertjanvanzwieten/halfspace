all: libokada.so

libokada.so: libokada.o
	gcc -shared -Wl,-soname,$@ -o $@ $< -lc

libokada.o: libokada.c
	gcc -fpic -g -c -o $@ -O3 -march=native -mtune=native $<

clean:
	rm -r -f build libokada.o libokada.so

.PHONY: all clean

# vim:noexpandtab
