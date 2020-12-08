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

using std::cout;
using std::cerr;
using std::endl;
using std::cin;

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
	size_t x = (int)p[0];
	size_t y = (int)p[1];

	if (x < 0 or x >= IM_X or y < 0 or y >= IM_Y)
		return Vector2f::Zero();

	if (x + 1 < IM_X)
		if (y + 1 < IM_Y)
			return lerp(
				p[1] - y,
				lerp(
					p[0] - x,
					vecs.block<2, 1>(0, y * IM_X + x),
					vecs.block<2, 1>(0, y * IM_X + x + 1)
				), lerp(
					p[0] - x,
					vecs.block<2, 1>(0, (y + 1) * IM_X + x),
					vecs.block<2, 1>(0, (y + 1) * IM_X + x + 1)
				)
			);
		else
			return lerp(
				p[1] - y,
				lerp(
					p[0] - x,
					vecs.block<2, 1>(0, y * IM_X + x),
					vecs.block<2, 1>(0, y * IM_X + x + 1)
				),
				Vector2f::Zero()
			);
	else if (y + 1 < IM_Y)
		return lerp(
			p[1] - y,
			lerp(
				p[0] - x,
				vecs.block<2, 1>(0, y * IM_X + x),
				Vector2f::Zero()
			), lerp(
				p[0] - x,
				vecs.block<2, 1>(0, (y + 1) * IM_X + x),
				Vector2f::Zero()
			)
		);
	else
		return lerp(
			p[1] - y,
			lerp(
				p[0] - x,
				vecs.block<2, 1>(0, y * IM_X + x),
				Vector2f::Zero()
			),
			Vector2f::Zero()
		);

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

int main(int argc, char **argv) {
	if (argc != 3) {
		printf("Usage: ./main image output\n");
		return 0;
	}
	int fd = checkLinux(open(argv[1], O_RDONLY));
	float *im = (float*)mmap(NULL, IM_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
	checkLinux((int)(size_t)im);
	close(fd);
	Vectors vecs(im);
	float *output = (float*)malloc(N_X * N_Y * STEPS * 2 * sizeof(float));
	checkLinux((int)(size_t)output);
	Lines out(output); 
    integrate(out, vecs);
    munmap(im, IM_SIZE);
    writeCSV(argv[2], out);
    free(output);
    return 0;
}