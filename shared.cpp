#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <time.h>
#include <sys/time.h>

using std::cout;
using std::cerr;
using std::endl;
using std::cin;
using std::vector;

using namespace Eigen;

constexpr size_t IM_X = 1300;
constexpr size_t IM_Y = 600;
constexpr size_t IM_V = 2 * sizeof(float);
constexpr size_t IM_SIZE = IM_X * IM_Y * IM_V;
constexpr size_t SR_X = 10;
constexpr size_t SR_Y = 5;
constexpr size_t N_X = IM_X / SR_X;
constexpr size_t N_Y = IM_Y / SR_Y;
constexpr size_t STEPS = 100;
constexpr float  dt = 1;

typedef Map<Matrix<float, 2, IM_X * IM_Y>> Vectors;
typedef Map<Matrix<float, 2, N_X * N_Y * STEPS>> Lines;

Vector2f lerp(float i, Ref<const Vector2f> a, Ref<const Vector2f> b) {
    return a + i * (b - a);
}

Vector2f biLinIntrp(Vectors& vecs, Ref<const Vector2f> p) {
    size_t x = (size_t)p[0];
    size_t y = (size_t)p[1];
    size_t x1 = x + 1;
    size_t y1 = y + 1;

    bool up = y >= 0;
    bool dn = y1 < IM_Y;
    bool lf = x >= 0;
    bool rt = x1 < IM_X;
    auto a = (lf and up) ? vecs.block<2, 1>(0, y  * IM_X + x ).eval() : Vector2f::Zero();
    auto b = (rt and up) ? vecs.block<2, 1>(0, y  * IM_X + x1).eval() : Vector2f::Zero();
    auto c = (lf and dn) ? vecs.block<2, 1>(0, y1 * IM_X + x ).eval() : Vector2f::Zero();
    auto d = (rt and dn) ? vecs.block<2, 1>(0, y1 * IM_X + x1).eval() : Vector2f::Zero();
    
    float delta = p[0] - x;
    a = lerp(delta, a, b);
    b = lerp(delta, c, d);
    return lerp(p[1] - y, a, b);
}

void integrate(Lines& out, Vectors& vecs) {
#pragma omp parallel for
    for (size_t i = 0; i < N_X * N_Y; i++) {
        size_t x = (i % N_X) * SR_X;
        size_t y = (i / N_X) * SR_Y;

        Vector2f k1, k2, k3, k4, q;
        Vector2f p(x, y);

        // Output location
        size_t idx = i * STEPS;
        
        // Initial output
        out.block<2, 1>(0, idx++) = p;

        // Integrate forward
        for (size_t i = 1; i < STEPS; i++) {
            k1 = dt * biLinIntrp(vecs, p);
            q = p + 0.5 * k1;
            k2 = dt * biLinIntrp(vecs, q);
            q = p + 0.5 * k2;
            k3 = dt * biLinIntrp(vecs, q);
            q = p + k3;
            k4 = dt * biLinIntrp(vecs, q);
            p += (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4);
            out.block<2, 1>(0, idx++) = p;
        }
    }
}

int checkLinux(int result) {
    if (result == -1) {
        cerr << "Linux Runtime Error: (" << errno << ") " << strerror(errno) << endl;
        abort();
    }
    return result;
}

void writeCSV(char* file, Lines& out) {
    const size_t file_size = N_X * N_Y * STEPS * (20 + 9 + 9 + 3);
    umask(0111);
    int fd = checkLinux(open(file, O_RDWR | O_CREAT | O_TRUNC, 06666));
    checkLinux(ftruncate(fd, file_size));
    char* map = (char*) mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, fd, 0);
    checkLinux((int)(size_t)map);
    char* cur = map;

    const char* header = "line_id, coordinate_x, coordinate_y\n";
    checkLinux(write(fd, header, strlen(header)));
    for (size_t i = 0; i < N_X * N_Y; i++)
        for (size_t s = 0; s < STEPS; s++) {
            Vector2f p = out.block<2, 1>(0, i * STEPS + s);
            cur += sprintf(cur, "%llu,%.7f,%.7f\n", i, p[0], p[1]);
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

static inline void stime(const char* name) {
    timespec cur_wall, cur_proc;
    clock_gettime(CLOCK_REALTIME, &cur_wall);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cur_proc);
    names.push_back(name);
    levels.push_back(cur_level++);
    wall.push_back(cur_wall);
    proc.push_back(cur_proc);
}

static inline void ftime() {
    timespec cur_wall, cur_proc;
    clock_gettime(CLOCK_REALTIME, &cur_wall);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cur_proc);
    levels.push_back(--cur_level);
    wall.push_back(cur_wall);
    proc.push_back(cur_proc);
}

// from https://gist.github.com/diabloneo/9619917
static inline void timespecDiff(timespec& a, timespec& b, timespec& result) {
    result.tv_sec  = a.tv_sec  - b.tv_sec;
    result.tv_nsec = a.tv_nsec - b.tv_nsec;
    if (result.tv_nsec < 0) {
        --result.tv_sec;
        result.tv_nsec += 1000000000L;
    }
}

static inline double timespecToMs(const timespec& t) {
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}

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

int main(int argc, char **argv) {
    stime("Program");
    stime("Setup");
    if (argc != 3) {
        ftime();
        ftime();
        printf("Usage: ./main image output\n");
        return 0;
    }

    stime("Read input");
    int fd = checkLinux(open(argv[1], O_RDONLY));
    float *im = (float*)mmap(NULL, IM_SIZE, PROT_READ, MAP_PRIVATE | MAP_LOCKED | MAP_POPULATE, fd, 0);
    checkLinux((int)(size_t)im);
    close(fd);
    ftime();

    Vectors vecs(im);
    float *output = (float*)malloc(N_X * N_Y * STEPS * 2 * sizeof(float));
    checkLinux((int)(size_t)output);
    Lines out(output);
    ftime();

    stime("Computation");
    integrate(out, vecs);
    ftime();

    //stime("Write");
    //writeCSV(argv[2], out);
    //ftime();

    stime("Free memory");
    munmap(im, IM_SIZE);
    free(output);
    ftime();

    ftime();

    ptime("Shared");

    return 0;
}