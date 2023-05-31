#include <stdio.h>
#include "gpu_containers.h"

__global__ void printPolygon(GPUPolygon &poly)
{
    printf("Hello from the GPU!\n");
    poly.print();
}

int main()
{
    GPUPoint points[3];

    for (int i = 0; i < 3; i++)
    {
        points[i] = GPUPoint(i + 1, i + 1);
    }

    GPUPolygon poly1 = GPUPolygon(3, points);
    // poly1.print();

    // Copy polygon to device
    GPUPolygon *polyD;
    cudaMalloc((void **)&polyD, sizeof(GPUPolygon));
    cudaMemcpy(polyD, &poly1, sizeof(GPUPolygon), cudaMemcpyHostToDevice);

    // Copy points to device
    GPUPoint *pointsD;
    cudaMalloc((void **)&pointsD, poly1.size * sizeof(GPUPoint));
    cudaMemcpy(pointsD, poly1.points, poly1.size * sizeof(GPUPoint), cudaMemcpyHostToDevice);

    // Set device polygon points pointer to copied points
    cudaMemcpy(&(polyD->points), &pointsD, sizeof(GPUPoint *), cudaMemcpyHostToDevice);

    printPolygon<<<1, 1>>>(*polyD);
    cudaDeviceSynchronize();

    cudaFree(polyD);
    cudaFree(pointsD);

    return 0;
}