PROJECT_NAME = heatSim

GCC := g++
NVCC := nvcc
RUN := srun -n 1 --gres=gpu:1

.PHONY: all
all: build
build: $(PROJECT_NAME)

heat.o:heat.cu util.h
	$(NVCC) $(CPPFLAGS) -c $<
$(PROJECT_NAME):heat.o 
	$(NVCC) -o $@ $<

run: build 
	$(RUN) $(PROJECT_NAME)

.PHONY: clean
clean:
	rm -f *.o $(PROJECT_NAME) *.dat
