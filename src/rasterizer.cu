#include <stdio.h>
#include <math.h>
#include "gpu_containers.h"

/**
 *  GLOBAL CONSTANTS
 */
// Rasterization cell marks
#define EMPTY_COLOR 0
#define PARTIAL_COLOR 1
#define FULL_COLOR 2
#define UNCERTAIN_COLOR 3
#define FULL_CHECKED 4

// Sample min/max
const GPUPoint S_MAX = GPUPoint(-66.8854, 49.3844);
const GPUPoint S_MIN = GPUPoint(-124.849, 24.5214);
// Hilbert space mix/max
const GPUPoint H_MAX = GPUPoint(65535, 65535); // 65535 = 2^16 - 1
const GPUPoint H_MIN = GPUPoint(0, 0);

__global__ void printPolygon(GPUPolygon &poly)
{
    printf("Hello from the GPU!\n");
    poly.print();
    poly.printMatrix();
}

__global__ void preparePolygonMatrix(GPUPolygon &poly, int mSize)
{
    // ID of current thread in reference to all created threads in kernel
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The number of threads in kernel
    int stride = blockDim.x * gridDim.x;

    // Create polygon rasterization matrix using a grid-stride loop
    for (int i = index; i < mSize; i += stride)
    {
        int row = (i / poly.mbrHeight) * poly.mbrWidth;
        poly.matrix[row + (i % poly.mbrWidth)] = UNCERTAIN_COLOR;
    }
}

__global__ void rasterizeBorder()
{
}

void CUDARasterize(GPUPolygon &poly)
{
    // Pointers used by the device.
    GPUPolygon *poly_D;
    GPUPoint *points_D;
    int *matrix_D;

    // Copy polygon to device
    cudaMalloc((void **)&poly_D, sizeof(GPUPolygon));
    cudaMemcpy(poly_D, &poly, sizeof(GPUPolygon), cudaMemcpyHostToDevice);

    // Copy points to device
    cudaMalloc((void **)&points_D, poly.size * sizeof(GPUPoint));
    cudaMemcpy(points_D, poly.points, poly.size * sizeof(GPUPoint), cudaMemcpyHostToDevice);

    // Copy polygon rasterization matrix to device
    size_t mSize = poly.mbrHeight * poly.mbrWidth * sizeof(int);
    cudaMalloc((void **)&matrix_D, mSize);
    cudaMemcpy(matrix_D, poly.matrix, mSize, cudaMemcpyHostToDevice);

    // Set device polygon's pointers to copied points & matrix
    cudaMemcpy(&(poly_D->points), &points_D, sizeof(GPUPoint *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(poly_D->matrix), &matrix_D, sizeof(int *), cudaMemcpyHostToDevice);

    preparePolygonMatrix<<<1, 3>>>(*poly_D, poly.mbrHeight * poly.mbrWidth);
    printPolygon<<<1, 1>>>(*poly_D);

    cudaDeviceSynchronize();

    cudaFree(poly_D);
    cudaFree(points_D);
    cudaFree(matrix_D);
}

void calculateMBR(GPUPolygon &poly)
{
    poly.hMax = GPUPoint(poly.points[0].x, poly.points[0].y);
    poly.hMin = GPUPoint(poly.points[0].x, poly.points[0].y);

    for (int i = 1; i < poly.size - 1; i++)
    {
        if (poly.points[i].x < poly.hMin.x)
        {
            poly.hMin.x = poly.points[i].x;
        }
        if (poly.points[i].y < poly.hMin.y)
        {
            poly.hMin.y = poly.points[i].y;
        }
        if (poly.points[i].x > poly.hMax.x)
        {
            poly.hMax.x = poly.points[i].x;
        }
        if (poly.points[i].y > poly.hMax.y)
        {
            poly.hMax.y = poly.points[i].y;
        }
    }

    // Round MBR and add 1 cell buffer around
    poly.hMin.x = floor(poly.hMin.x) - 1;
    poly.hMin.y = floor(poly.hMin.y) - 1;
    poly.hMax.x = ceil(poly.hMax.x) + 1;
    poly.hMax.y = ceil(poly.hMax.y) + 1;

    poly.mbrWidth = poly.hMax.x - poly.hMin.x;
    poly.mbrHeight = poly.hMax.y - poly.hMin.y;
}

int main(void)
{
    // Create a test square
    GPUPoint testPoints[5];
    testPoints[0] = GPUPoint(2, 2);
    testPoints[1] = GPUPoint(12, 22);
    testPoints[2] = GPUPoint(32, 32);
    testPoints[3] = GPUPoint(22, 12);
    testPoints[4] = GPUPoint(2, 2);

    GPUPolygon testPoly = GPUPolygon(5, testPoints);
    calculateMBR(testPoly);
    // testPoly.mbrWidth = 3;
    // testPoly.mbrHeight = 3;
    testPoly.matrix = new int[testPoly.mbrWidth * testPoly.mbrHeight];
    
    testPoly.print();
    testPoly.printMatrix();

    CUDARasterize(testPoly);

    return 0;
}