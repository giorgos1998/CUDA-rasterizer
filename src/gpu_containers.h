#ifndef GPU_CONTAINERS_H
#define GPU_CONTAINERS_H

class GPUPoint
{
public:
    float x;
    float y;

    __host__ __device__ GPUPoint(float x, float y);
    __host__ __device__ GPUPoint();
    __host__ __device__ GPUPoint(GPUPoint &that);
    __host__ __device__ ~GPUPoint();
    __host__ __device__ void print();
};

class GPUPolygon
{
public:
    // NOTE: 1st point is the same as the last one.
    // Normalized points based on MBR.
    GPUPoint *points;
    // NOTE: 1st point is the same as the last one.
    // Points in Hilbert space.
    GPUPoint *hilbertPoints;
    int size;
    // Hilbert space polygon rasterization matrix
    int *matrix;
    // Hilbert space polygon min/max (MBR)
    GPUPoint hMin;
    GPUPoint hMax;
    int mbrWidth;
    int mbrHeight;

    __host__ __device__ GPUPolygon(int size, GPUPoint points[]);
    __host__ __device__ GPUPolygon(GPUPolygon &that);
    __host__ __device__ GPUPolygon& operator=(const GPUPolygon &that);
    __host__ __device__ ~GPUPolygon();
    __host__ __device__ int getMatrixXY(int x, int y);
    __host__ __device__ void setMatrixXY(int x, int y, int value);
    __host__ __device__ void print();
    __host__ __device__ void printMatrix();
};

struct GPUStackItem
{
    GPUPoint point;
    GPUStackItem *prevItem;
};

class GPUStack
{
public:
    int size;
    GPUStackItem *lastItem;

    __host__ __device__ GPUStack();
    __host__ __device__ ~GPUStack();
    __host__ __device__ GPUPoint pop();
    __host__ __device__ bool hasItems();
    __host__ __device__ void push(int x, int y);
    __host__ __device__ void print();
};

#endif