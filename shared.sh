#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH -o slurmjob-%j.out-%N 
#SBATCH -e slurmjob-%j.err-%N 
#SBATCH --account=usucs5890 
#SBATCH --partition=kingspeak

# Load MPI implementation library
module load gcc/6.4.0 eigen/3.3.7

# Compile application
g++ -O2 -fopenmp -I$EIGENROOT/include -o shared shared.cpp

echo "running with 64 threads"
export OMP_NUM_THREADS=64
./dist vecs.raw out.csv

echo "running with 32 threads"
export OMP_NUM_THREADS=32
./dist vecs.raw out.csv

echo "running with 16 threads"
export OMP_NUM_THREADS=16
./dist vecs.raw out.csv

echo "running with 4 threads"
export OMP_NUM_THREADS=4
./dist vecs.raw out.csv

echo "running with 1 threads"
export OMP_NUM_THREADS=1
./dist vecs.raw out.csv
