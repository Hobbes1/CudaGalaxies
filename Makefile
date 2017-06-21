CC = g++
NVCC = nvcc
CFLAGS = -std=c++11
NVFLAGS = -arch=sm_30
LIBS = -lpthread

nbody: nBodyCuda.o main.o 
	$(NVCC) $(NVFLAGS) -o nbody main.o $(LIBS)

main.o: main.cu 
	$(NVCC) $(NVFLAGS) -c $< $(LIBS)

nBodyCuda.o: nBodyCuda.cu nBodyCuda.h
	$(NVCC) $(NVFLAGS) -c $<  $(LIBS)

clean:
	rm -f *.o 
	
