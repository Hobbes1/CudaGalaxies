CUDA_FILES := $(wildcard src/*.cu)
OBJ_FILES := $(addprefix obj/,$(notdir $(CUDA_FILES:.cu=.o)))

NVCC = nvcc
NVFLAGS = -arch=sm_30
LIBS = -lpthread

nbody: $(OBJ_FILES) 
	$(NVCC) $(NVFLAGS) -o nbody main.o $(LIBS)

obj/%.o: src/%.cu
	$(NVCC) $(NVFLAGS) -c $< $(LIBS)

clean:
	rm -f *.o 
	mv *.data *Params data
	
