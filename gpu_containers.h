#ifndef GPU_CONTAINERS_H
#define GPU_CONTAINERS_H

class GPUPoint
{
public:
    float x;
    float y;

    __host__ __device__ GPUPoint(float x, float y);
    __host__ __device__ GPUPoint();
    __host__ __device__ ~GPUPoint();
    __host__ __device__ void print();
};

class GPUPolygon
{
public:
    GPUPoint *points;
    int size;

    __host__ __device__ GPUPolygon(int size, GPUPoint points[]);
    __host__ __device__ GPUPolygon(GPUPolygon &that);
    __host__ __device__ GPUPolygon& operator=(const GPUPolygon& that);
    __host__ __device__ ~GPUPolygon();
    __host__ __device__ void print();
};

#endif