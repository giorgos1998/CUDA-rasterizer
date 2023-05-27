#include "gpu_containers.h"

int main()
{
    GPUPoint points[3];
    GPUPoint newPoints[3];

    for (int i = 0; i < 3; i++)
    {
        points[i] = GPUPoint(i+1, i+1);
        newPoints[i] = GPUPoint(i+2, i+2);
    }

    GPUPolygon poly1 = GPUPolygon(3, points);
    poly1.print();

    GPUPolygon poly2 = GPUPolygon(3, newPoints);
    poly2.print();

    poly1 = poly2;
    poly1.print();
    
    return 0;
}