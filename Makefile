# On Trestles we will check versus your performance versus Intel MKL library's BLAS. 

CC = icc 
OPT = -O3 -funroll-loops -ftree-vectorize
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/apps/intel/15/composer_xe_2015.2.164/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm


targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-hodges
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-hodges.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-hodges.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout
