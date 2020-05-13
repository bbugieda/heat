CC = cc -g -pedantic -std=c11
MPICC = mpicc -pedantic -std=c11
DEPS=settings.h

HOST := $(shell hostname)
ifeq ($(HOST),cisc372)
RUN = srun
else
RUN = mpiexec
endif

nomp_noconv: mpi_nomp_noconv.c
	$(MPICC) -o nomp_noconv mpi_nomp_noconv.c -lm -fopenmp

nomp: mpi_nomp.c
	$(MPICC) -o nomp mpi_nomp.c -lm -fopenmp

collapse_noconv4: mpi_collapse_noconv4.c
	$(MPICC) -o collapse_noconv4 mpi_collapse_noconv4.c -lm -fopenmp

collapse4: mpi_collapse4.c
	$(MPICC) -o collapse4 mpi_collapse4.c -lm -fopenmp

omp: mpi_omp.c
	$(MPICC) -o omp mpi_omp.c -lm -fopenmp


all: nomp_noconv nomp omp collapse_noconv4 collapse4

clean: 
	rm -f omp nomp nomp_noconv collapse_noconv4 collapse4

