gpu: gpu.cu
	nvcc -O2 gpu.cu -o gpu
	./gpu vecs.raw out.csv
	python csv_to_vtp.py out.csv
	paraview out.csv.vtp

shared: shared.cpp
	g++ -g -fopenmp shared.cpp -o shared
	./shared vecs.raw out.csv
	python csv_to_vtp.py out.csv
	paraview out.csv.vtp
