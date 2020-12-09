#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH -o slurmjob-%j.out-%N 
#SBATCH -e slurmjob-%j.err-%N 
#SBATCH --account=usucs5890 
#SBATCH --partition=kingspeak

# Load MPI implementation library
module load gcc/6.4.0 mpich/3.3 eigen/3.3.7

# Compile application
mpic++ -O2 -I$EIGENROOT/include -o dist dist.cpp

echo "running with 64 cores"
mpirun -np 64 ./dist vecs.raw out.csv

echo "running with 32 cores"
mpirun -np 32 ./dist vecs.raw out.csv

echo "running with 16 cores"
mpirun -np 16 ./dist vecs.raw out.csv

echo "running with 4 cores"
mpirun -np 4 ./dist vecs.raw out.csv

echo "running with 1 core"
mpirun -np 1 ./dist vecs.raw out.csv
