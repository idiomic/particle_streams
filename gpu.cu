#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>
#include <vector>
#include <time.h>
#include <sys/time.h>

using std::cout;
using std::cerr;
using std::endl;
using std::cin;
using std::vector;

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

vector<const char*> names;
vector<timespec> wall;
vector<timespec> proc;
vector<size_t> levels;
size_t cur_level = 0;

__host__
static inline void stime(const char* name) {
    timespec cur_wall, cur_proc;
    clock_gettime(CLOCK_REALTIME, &cur_wall);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cur_proc);
    names.push_back(name);
    levels.push_back(cur_level++);
    wall.push_back(cur_wall);
    proc.push_back(cur_proc);
}

__host__
static inline void ftime() {
    timespec cur_wall, cur_proc;
    clock_gettime(CLOCK_REALTIME, &cur_wall);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cur_proc);
    levels.push_back(--cur_level);
    wall.push_back(cur_wall);
    proc.push_back(cur_proc);
}

// from https://gist.github.com/diabloneo/9619917
__host__
static inline void timespecDiff(timespec& a, timespec& b, timespec& result) {
    result.tv_sec  = a.tv_sec  - b.tv_sec;
    result.tv_nsec = a.tv_nsec - b.tv_nsec;
    if (result.tv_nsec < 0) {
        --result.tv_sec;
        result.tv_nsec += 1000000000L;
    }
}

__host__
static inline double timespecToMs(const timespec& t) {
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}

__host__
static size_t ptime(const char* name, size_t n = 0, size_t i = 0, size_t l = 0) {
    while (n < names.size() and levels[i] == l) {
        size_t j = i + 1;
        auto& sw = wall[i];
        auto& sp = proc[i];
        int jumped = j;
        while (l < levels[j]) j++;
        auto& fw = wall[j];
        auto& fp = proc[j];
        timespec w, p;
        timespecDiff(fw, sw, w);
        timespecDiff(fp, sp, p);
        for (size_t k = 0; k < l; k++)
            printf("\t");
        printf("\"%s\", \"%s\", %.3f, %.3f\n",
            name,
            names[n++],
            timespecToMs(w),
            timespecToMs(p));
        if (jumped < j)
            n = ptime(name, n, jumped, l + 1);
        i = j + 1;
    }
    return n;
}

__host__
int main(int argc, char **argv) {
    stime("Program");
    stime("Setup");
    if (argc != 3) {
        ftime();
        ftime();
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
    stime("Read input");
    int fd = checkLinux(open(argv[1], O_RDONLY));

    // Allocating + Mapping host memory
    float2 *im;
    cudaArray* im_d;
    float2 *output_d;
    float2 *output;

    // Memory mapping does not provide a performance boost.
    // It trades off between copy time to GPU or copy to RAM.
    checkCuda(cudaMallocHost(&im, IM_SIZE));
    checkLinux(read(fd, im, IM_SIZE));
    close(fd);
    ftime();

    // Modified basic cuda texture manipulation obtained from
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

    // Allocate CUDA array in device memory
    stime("Copy to GPU");
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0,
        cudaChannelFormatKindFloat);
    checkCuda(cudaMallocArray(&im_d, &channelDesc, IM_X, IM_Y));
    checkCuda(cudaMemcpyToArray(im_d, 0, 0, im, IM_SIZE, cudaMemcpyHostToDevice));
    ftime();

    // Specify texture
    stime("Initialize Texture");
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
    texDesc.maxAnisotropy    = 2;
    texDesc.normalizedCoords = false;

    // Create texture object
    cudaTextureObject_t imTex = 0;
    checkCuda(cudaCreateTextureObject(&imTex, &resDesc, &texDesc, NULL));
    ftime();

    dim3 block(26, 24, 1);
    dim3 grid(5, 5, 1);
    // dim3 block(1, 24, 1);
    // dim3 grid(1, 25, 1);
    const size_t num_particles = block.x * grid.x * block.y * grid.y;
    const size_t out_size = num_particles * sizeof(float2) * steps;

    stime("Allocate Output");
    checkCuda(cudaMalloc(&output_d, out_size));
    ftime();

    ftime();

    stime("Computation");
    integrate<<<grid, block>>>(output_d, imTex, dt, steps);
    ftime();

    // Copying from device to host
    stime("Copy to host");
    checkCuda(cudaMallocHost(&output, out_size));
    checkCuda(cudaMemcpy(output, output_d, out_size, cudaMemcpyDeviceToHost));
    ftime();

    stime("Free device memory");
    checkCuda(cudaFree(output_d));
    checkCuda(cudaDestroyTextureObject(imTex));
    checkCuda(cudaFreeArray(im_d));
    ftime();

    //stime("Write");
    //writeCSV(argv[2], output, num_particles, steps);
    //ftime();

    stime("Free host memory");
    checkCuda(cudaFreeHost(im));
    checkCuda(cudaFreeHost(output));
    ftime();

    ftime();

    ptime("GPU");

    return 0;
}