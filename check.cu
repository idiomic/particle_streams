int main() {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

#define PRINT_PROP(prop_name)\
	cout << #prop_name << " " << props.prop_name << "\n"

#define DIM3(prop_name)\
	PRINT_PROP(prop_name[0]);\
	PRINT_PROP(prop_name[1]);\
	PRINT_PROP(prop_name[2])

#define DIM2(prop_name)\
	PRINT_PROP(prop_name[0]);\
	PRINT_PROP(prop_name[1])

	PRINT_PROP(name);
	PRINT_PROP(totalGlobalMem);
	PRINT_PROP(sharedMemPerBlock);
	PRINT_PROP(regsPerBlock);
	PRINT_PROP(maxThreadsPerBlock);
	PRINT_PROP(totalConstMem);
	PRINT_PROP(multiProcessorCount);
	DIM3(maxThreadsDim);
	DIM3(maxGridSize);
	DIM2(maxTexture2D);
	DIM2(maxTexture2DMipmap);
	DIM3(maxTexture2DLinear);
	return 0;
}