CC = gcc

CFLAGS= -fopenmp -fPIC -O3 -D NDEBUG -Wall -Werror
LIBS = -lgsl -lgslcblas -lm -lgomp

all: librwalk.so

librwalk.so: rwalk.c
	$(CC) -shared -o librwalk.so $(CFLAGS) rwalk.c $(LIBS)
clean:
	rm -rf librwalk.so __pycache__
