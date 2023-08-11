#include <stdio.h>
#include <math.h>
#include <assert.h>
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
        // poly.matrix[i] = i + 1;
        poly.matrix[i] = UNCERTAIN_COLOR;
    }
}

__global__ void rasterizeBorder(GPUPolygon &poly)
{
    GPUPoint startPoint, endPoint;
    GPUPoint startCell, endCell;
    GPUPoint step;
    int nextVertical, nextHorizontal;
    float gradient, edgeLength;
    GPUPoint tMax, tDelta;

    // ID of current thread in reference to all created threads in kernel.
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;

    // Number of threads MUST be equal to the edges of the polygon.
    // TODO improve a bit
    assert(t_ID < poly.size);

    // Find edge stard/end with positive orientation (on x axis).
    if (poly.points[t_ID].x < poly.points[t_ID + 1].x)
    {
        startPoint = GPUPoint(poly.points[t_ID]);
        endPoint = GPUPoint(poly.points[t_ID + 1]);
    }
    else
    {
        startPoint = GPUPoint(poly.points[t_ID + 1]);
        endPoint = GPUPoint(poly.points[t_ID]);
    }
    // printf("Edge %d\n", t_ID + 1);
    // printf("Starting point: ");
    // startPoint.print();
    // printf("Ending point: ");
    // endPoint.print();

    startCell.x = (int)startPoint.x;
    startCell.y = (int)startPoint.y;
    endCell.x = (int)endPoint.x;
    endCell.y = (int)endPoint.y;
    // printf("Starting cell: ");
    // startCell.print();
    // printf("Ending cell: ");
    // endCell.print();

    // Edge always goes from smaller X to larger.
    step.x = 1;
    step.y = endPoint.y > startPoint.y ? 1 : -1;
    // printf("Steps: ");
    // step.print();

    // Find nearest vertical & horizontal grid lines based on edge direction.
    nextVertical = ceilf(startPoint.x);
    nextHorizontal = step.y == 1 ? ceilf(startPoint.y) : floorf(startPoint.y);
    // printf("Next horizontal: %d, next vertical: %d\n", nextHorizontal, nextVertical);

    gradient = (endPoint.y - startPoint.y) / (endPoint.x - startPoint.x);
    edgeLength = sqrtf(powf(endPoint.x - startPoint.x, 2) + powf(endPoint.y - startPoint.y, 2));

    // Find intersection with nearest vertical & find tMax.
    float intersectY = startPoint.y + (gradient * (nextVertical - startPoint.x));
    tMax.x = sqrtf(powf(nextVertical - startPoint.x, 2) + powf(intersectY - startPoint.y, 2));

    // Find intersection with nearest horizontal & find tMax.
    float intersectX = ((nextHorizontal - startPoint.y) / gradient) + startPoint.x;
    tMax.y = sqrtf(powf(intersectX - startPoint.x, 2) + powf(nextHorizontal - startPoint.y, 2));

    // printf("Gradient: %f\n", gradient);
    // printf("Intersection with vertical at: (%d, %f)\n", nextVertical, intersectY);
    // printf("Intersection with horizontal at: (%f, %d)\n", intersectX, nextHorizontal);
    // printf("tMax: ");
    // tMax.print();

    // TODO check length of line to see if the intersection point is in range.

    tDelta.x = edgeLength / (endPoint.x - startPoint.x);
    tDelta.y = edgeLength / fabsf(endPoint.y - startPoint.y);

    // Edge traversal, we traverse using the startPoint to save memory.
    while (startPoint.x < endCell.x + 1 && startPoint.y < endCell.y + step.y)
    {
        poly.matrix[((int)startPoint.y * poly.mbrWidth) + (int)startPoint.x] = PARTIAL_COLOR;
        // printf("Painted (%d, %d)\n", (int)startPoint.x, (int)startPoint.y);

        if (tMax.x < tMax.y)
        {
            startPoint.x += step.x;
            tMax.x += tDelta.x;
        }
        else
        {
            startPoint.y += step.y;
            tMax.y += tDelta.y;
        }
    }
}

__global__ void fillPolygonPerPixel(GPUPolygon &poly, int matrixSize)
{
    GPUPoint edgeStart, edgeEnd;
    GPUPoint testPoint;

    // ID of current thread in reference to all created threads in kernel
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The number of threads in kernel
    int stride = blockDim.x * gridDim.x;

    // Run through the whole rasterization matrix using a grid-stride loop
    for (int pixelID = index; pixelID < matrixSize; pixelID += stride)
    {
        if (poly.matrix[pixelID] == PARTIAL_COLOR) { continue; }

        bool isInside = false;
        // int intersections = 0;

        // Find current pixel coordinates
        testPoint.x = pixelID % poly.mbrWidth;
        testPoint.y = pixelID / poly.mbrWidth;

        // Loop all edges
        for (int j = 0; j < poly.size-1; j++)
        {
            // j = i (start)
            // i = j+1 (end)
            edgeStart = poly.points[j];
            edgeEnd = poly.points[j+1];

            // Check intersection with current edge
            if (((edgeEnd.y > testPoint.y) != (edgeStart.y > testPoint.y)) &&
                (testPoint.x < (edgeStart.x - edgeEnd.x) * (testPoint.y - edgeStart.y) /
                (edgeStart.y - edgeEnd.y) + edgeStart.x))
            {
                isInside = !isInside;
                // intersections++;
            }
        }

        poly.matrix[pixelID] = (isInside) ? FULL_COLOR : EMPTY_COLOR;
        // if (pixelID == 29) 
        // {
        //     printf("Pixel %d intersections: %d\n", pixelID, intersections);
        //     testPoint.print();
        // }
    }
}

__global__ void floodFillPolygonInSectors(GPUPolygon &poly, int xSectors, int ySectors)
{
    GPUPoint sectorSize;

    sectorSize.x = ceilf(poly.mbrWidth / xSectors);
    sectorSize.y = ceilf(poly.mbrHeight / ySectors);

    
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

    preparePolygonMatrix<<<1, 1024>>>(*poly_D, poly.mbrHeight * poly.mbrWidth);
    rasterizeBorder<<<1, 4>>>(*poly_D);
    fillPolygonPerPixel<<<1, 10>>>(*poly_D, poly.mbrHeight * poly.mbrWidth);
    printPolygon<<<1, 1>>>(*poly_D);

    cudaDeviceSynchronize();

    cudaFree(poly_D);
    cudaFree(points_D);
    cudaFree(matrix_D);
}

void calculateMBR(GPUPolygon &poly)
{
    poly.hMax = GPUPoint(poly.hilbertPoints[0].x, poly.hilbertPoints[0].y);
    poly.hMin = GPUPoint(poly.hilbertPoints[0].x, poly.hilbertPoints[0].y);

    for (int i = 1; i < poly.size - 1; i++)
    {
        if (poly.hilbertPoints[i].x < poly.hMin.x)
        {
            poly.hMin.x = poly.hilbertPoints[i].x;
        }
        if (poly.hilbertPoints[i].y < poly.hMin.y)
        {
            poly.hMin.y = poly.hilbertPoints[i].y;
        }
        if (poly.hilbertPoints[i].x > poly.hMax.x)
        {
            poly.hMax.x = poly.hilbertPoints[i].x;
        }
        if (poly.hilbertPoints[i].y > poly.hMax.y)
        {
            poly.hMax.y = poly.hilbertPoints[i].y;
        }
    }

    // Round MBR and add 1 cell buffer around
    poly.hMin.x = (int)poly.hMin.x - 1;
    poly.hMin.y = (int)poly.hMin.y - 1;
    poly.hMax.x = (int)poly.hMax.x + 1;
    poly.hMax.y = (int)poly.hMax.y + 1;

    poly.mbrWidth = poly.hMax.x - poly.hMin.x + 1;
    poly.mbrHeight = poly.hMax.y - poly.hMin.y + 1;
}

void normalizePoints(GPUPolygon &poly)
{
    for (int i = 0; i < poly.size; i++)
    {
        poly.points[i] = GPUPoint(
            poly.hilbertPoints[i].x - poly.hMin.x,
            poly.hilbertPoints[i].y - poly.hMin.y);
    }
}

int main(void)
{
    // Create a test square
    GPUPoint testPoints[5];
    testPoints[0] = GPUPoint(1.5, 1.5);
    testPoints[1] = GPUPoint(11.5, 21.5);
    testPoints[2] = GPUPoint(31.5, 31.5);
    testPoints[3] = GPUPoint(21.5, 11.5);
    testPoints[4] = GPUPoint(1.5, 1.5);

    GPUPolygon testPoly = GPUPolygon(5, testPoints);
    calculateMBR(testPoly);
    // testPoly.mbrWidth = 5;
    // testPoly.mbrHeight = 3;
    normalizePoints(testPoly);
    testPoly.matrix = new int[testPoly.mbrWidth * testPoly.mbrHeight];

    // testPoly.print();
    // testPoly.printMatrix();

    // CUDARasterize(testPoly);
    // TODO move rasterization matrix to Hilbert space

    return 0;
}