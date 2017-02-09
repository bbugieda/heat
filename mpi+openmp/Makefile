CC=mpicc
DEPS=settings.h

nomp_noconv: mpi_nomp_noconv.c
	$(CC) -o nomp_noconv mpi_nomp_noconv.c -lm -fopenmp

nomp: mpi_nomp.c
	$(CC) -o nomp mpi_nomp.c -lm -fopenmp

collapse_noconv4: mpi_collapse_noconv4.c
	$(CC) -o collapse_noconv4 mpi_collapse_noconv4.c -lm -fopenmp

collapse4: mpi_collapse4.c
	$(CC) -o collapse4 mpi_collapse4.c -lm -fopenmp

omp: mpi_omp.c
	$(CC) -o omp mpi_omp.c -lm -fopenmp


all: nomp_noconv nomp omp collapse_noconv4 collapse4

clean: 
	rm -f omp nomp nomp_noconv collapse_noconv4 collapse4

