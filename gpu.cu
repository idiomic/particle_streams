#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>

using std::cout;
using std::cerr;
using std::endl;
using std::cin;

constexpr size_t IM_X = 1300;
constexpr size_t IM_Y = 600;
constexpr size_t IM_V = sizeof(float2);
constexpr size_t IM_SIZE = IM_X * IM_Y * IM_V;
constexpr size_t XSR = 10;
constexpr size_t YSR = 5;

__device__
inline float2 mul(float s, float2 v) {
	v.x *= s;
	v.y *= s;
	return v;
}

__device__
inline float2 add(float2 v1, float2 v2) {
	v1.x += v2.x;
	v1.y += v2.y;
	return v1;
}

__global__
void integrate(float2* out, cudaTextureObject_t vecs, float dt, size_t steps) {
    float2 k1, k2, k3, k4, p, q;

    // Initial position
    p.x = blockIdx.x * blockDim.x + threadIdx.x;
    p.y = blockIdx.y * blockDim.y + threadIdx.y;

    // Output location
    size_t idx = (blockDim.x * gridDim.x * (int)p.y + (int)p.x) * steps;

    // Apply sample rate
    p.x *= XSR;
    p.y *= YSR;

    // Initial output
    out[idx++] = p;

    // Integrate forward
    for (size_t i = 1; i < steps; i++) {
        k1 = mul(dt, tex2D<float2>(vecs, p.x, p.y));
        q = add(p, mul(0.5, k1));
        k2 = mul(dt, tex2D<float2>(vecs, q.x, q.y));
        q = add(p, mul(0.5, k2));
        k3 = mul(dt, tex2D<float2>(vecs, q.x, q.y));
        q = add(p, k3);
        k4 = mul(dt, tex2D<float2>(vecs, q.x, q.y));
        p.x += (1.0/6.0)*(k1.x + 2*k2.x + 2*k3.x + k4.x);
        p.y += (1.0/6.0)*(k1.y + 2*k2.y + 2*k3.y + k4.y);
        out[idx++] = p;
    }
}

__host__
cudaError_t checkCuda(cudaError_t result) {
	if (result != cudaSuccess) {
		cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
		abort();
	}
	return result;
}

__host__
int checkLinux(int result) {
	if (result == -1) {
		cerr << "Linux Runtime Error: (" << errno << ") " << strerror(errno) << endl;
		abort();
	}
	return result;
}

__host__
void writeCSV(char* file, float2* output, size_t num_particles, size_t steps) {
	const size_t file_size = num_particles * steps * (20 + 9 + 9 + 3);
	umask(0111);
	int fd = checkLinux(open(file, O_RDWR | O_CREAT | O_TRUNC, 06666));
	checkLinux(ftruncate(fd, file_size));
	char* map = (char*) mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
	checkLinux((int)(size_t)map);
	char* cur = map;

	const char* header = "line_id, coordinate_x, coordinate_y\n";
	checkLinux(write(fd, header, strlen(header)));
	for (size_t i = 0; i < num_particles; i++)
		for (size_t s = 0; s < steps; s++) {
			float2 p = output[i * steps + s];
			cur += sprintf(cur, "%llu,%.7f,%.7f\n", i, p.x, p.y);
		}
	msync(map, file_size, MS_SYNC);
	munmap(map, file_size);
	checkLinux(ftruncate(fd, cur - map));
	checkLinux(close(fd));
}

__host__
int main(int argc, char **argv) {
	if (argc != 3) {
		printf("Usage: ./main image output\n");
		return 0;
	}

	float dt = 1;
	//cout << "Enter delta time: ";
	//cin >> dt;

	size_t steps = 100;
	//cout << "Enter number of steps: ";
	//cin >> steps;

	// Opening file
	int fd = checkLinux(open(argv[1], O_RDONLY));

	// Allocating + Mapping host memory
	float2 *im;
    cudaArray* im_d;
    float2 *output_d;
    float2 *output;

	checkCuda(cudaMallocHost(&im, IM_SIZE));
	checkLinux(read(fd, im, IM_SIZE));
	close(fd);

    // Modified basic cuda texture manipulation obtained from
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

	// Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0,
        cudaChannelFormatKindFloat);
    checkCuda(cudaMallocArray(&im_d, &channelDesc, IM_X, IM_Y));
    checkCuda(cudaMemcpyToArray(im_d, 0, 0, im, IM_SIZE, cudaMemcpyHostToDevice));
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = im_d;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.maxAnisotropy	 = 2;
    texDesc.normalizedCoords = false;

    // Create texture object
    cudaTextureObject_t imTex = 0;
    checkCuda(cudaCreateTextureObject(&imTex, &resDesc, &texDesc, NULL));

	dim3 block(26, 24, 1);
	dim3 grid(5, 5, 1);
	// dim3 block(1, 24, 1);
	// dim3 grid(1, 25, 1);
	const size_t num_particles = block.x * grid.x * block.y * grid.y;
	const size_t out_size = num_particles * sizeof(float2) * steps;

    checkCuda(cudaMalloc(&output_d, out_size));

    integrate<<<grid, block>>>(output_d, imTex, dt, steps);

	// Copying from device to host
	checkCuda(cudaMallocHost(&output, out_size));
	checkCuda(cudaMemcpy(output, output_d, out_size, cudaMemcpyDeviceToHost));

    checkCuda(cudaDestroyTextureObject(imTex));
    checkCuda(cudaFreeArray(im_d));

    writeCSV(argv[2], output, num_particles, steps);

	checkCuda(cudaFreeHost(im));
    checkCuda(cudaFree(output_d));
    checkCuda(cudaFreeHost(output));

    return 0;
}