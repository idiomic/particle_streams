.PHONY: run_gpu run_shared run_dist\

run_gpu: gpu
	./gpu vecs.raw out.csv
	#python csv_to_vtp.py out.csv
	#paraview out.csv.vtp

run_shared: shared
	./shared vecs.raw out.csv
	#python csv_to_vtp.py out.csv
	#paraview out.csv.vtp

run_dist: dist
	mpirun -np 12 ./dist vecs.raw out.csv
	#python csv_to_vtp.py out.csv
	#paraview out.csv.vtp

shared: shared.cpp
	g++ -O2 -fopenmp shared.cpp -o shared

gpu: gpu.cu
	nvcc -O2 gpu.cu -o gpu

dist: dist.cpp
	mpic++ -g dist.cpp -o dist
