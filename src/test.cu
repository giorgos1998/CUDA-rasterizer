#include <stdio.h>
#include "gpu_containers.h"

__global__ void printPolygon(GPUPolygon poly)
{
    printf("Hello from the GPU!\n");
    poly.print();
}

int main()
{
    GPUPoint points[3];
    // GPUPoint newPoints[3];

    for (int i = 0; i < 3; i++)
    {
        points[i] = GPUPoint(i+1, i+1);
        // newPoints[i] = GPUPoint(i+2, i+2);
    }

    GPUPolygon poly1 = GPUPolygon(3, points);
    poly1.print();

    // GPUPolygon poly2 = GPUPolygon(3, newPoints);
    // poly2.print();

    printPolygon<<<1, 1>>>(poly1);
    cudaDeviceSynchronize();

    return 0;
}