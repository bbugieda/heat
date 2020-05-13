#!/bin/bash

pset="1 2 4"

echo "No OpenMP-No convergence:"
for p in ${pset} 
do
	echo "#####Processes=$p#####"
	srun -n $p nomp_noconv
done

echo "No OpenMP-With convergence:"
for p in ${pset} 
do
	echo "#####Processes=$p#####"
	srun -n $p nomp
done

echo "With 4 OpenMP threads-No convergence:"
for p in ${pset} 
do
	echo "#####Processes=$p#####"
	srun -n $p collapse_noconv4
done

echo "With 4 OpenMP threads-With convergence:"
for p in ${pset} 
do
	echo "#####Processes=$p#####"
	srun -n $p collapse4
done

echo "Super OpenMP threads"
for p in ${pset} 
do
	echo "#####Processes=$p#####"
	srun -n $p omp
done

