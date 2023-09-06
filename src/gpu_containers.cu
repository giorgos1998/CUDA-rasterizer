/**
 * @file gpu_containers.cu
 * @brief This file contains classes' functions used across the CUDA rasterizer.
 */

#include <stdio.h>
#include <assert.h>

#include "gpu_containers.cuh"
#include "constants.h"

__host__ __device__ GPUPoint::GPUPoint()
{
    this->x = 0;
    this->y = 0;
}

__host__ __device__ GPUPoint::GPUPoint(double x, double y)
{
    this->x = x;
    this->y = y;
}

__host__ __device__ GPUPoint::GPUPoint(const GPUPoint &that)
{
    this->x = that.x;
    this->y = that.y;
}

__host__ __device__ GPUPoint::~GPUPoint()
{
    // printf("Deleting point\n");
}

__host__ __device__ bool GPUPoint::operator==(const GPUPoint &that)
{
    if (this->x == that.x && this->y == that.y)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__host__ __device__ void GPUPoint::print(bool newLine)
{
    printf("(%7.3f, %7.3f)", x, y);
    if (newLine)
    {
        printf("\n");
    }
}

__host__ __device__ GPUPolygon::GPUPolygon(int id, int size, GPUPoint points[])
{
    this->id = id;
    this->size = size;
    this->hilbertPoints = new GPUPoint[size];
    this->points = new GPUPoint[size];

    for (int i = 0; i < size; i++)
    {
        this->hilbertPoints[i] = points[i];
    }
}

__host__ __device__ GPUPolygon::GPUPolygon(const GPUPolygon &that)
{
    // printf("In copy constructor\n");

    this->id = that.id;
    this->size = that.size;
    this->hilbertPoints = new GPUPoint[size];
    this->points = new GPUPoint[this->size];

    for (int i = 0; i < size; i++)
    {
        this->hilbertPoints[i] = that.hilbertPoints[i];
        this->points[i] = that.points[i];
    }
}

__host__ __device__ GPUPolygon &GPUPolygon::operator=(const GPUPolygon &that)
{
    // printf("In copy assignment operator\n");
    if (this != &that)
    {
        // Using assert to work both on host and device.
        // Stop execution if the polygons don't have the same size.
        assert(this->size == that.size);

        this->id = that.id;
        this->size = that.size;
        this->hilbertPoints = new GPUPoint[size];
        this->points = new GPUPoint[this->size];

        for (int i = 0; i < size; i++)
        {
            this->hilbertPoints[i] = that.hilbertPoints[i];
            this->points[i] = that.points[i];
        }
    }
    return *this;
}

__host__ __device__ GPUPolygon::~GPUPolygon()
{
    // printf("Deleting polygon\n");
}

__host__ __device__ int GPUPolygon::getMatrixXY(int x, int y)
{
    return matrix[y * mbrWidth + x];
}

__host__ __device__ void GPUPolygon::setMatrixXY(int x, int y, int value)
{
    matrix[y * mbrWidth + x] = value;
}

__host__ __device__ void GPUPolygon::print()
{
    printf("---------- Polygon %d ----------\n", id);
    printf("Hilbert min: ");
    hMin.print();
    printf("Hilbert max: ");
    hMax.print();
    printf("Matrix size (WxH): %dx%d\n", mbrWidth, mbrHeight);
    printf("Polygon size: %d\n", size);
    if (size > 20)
    {
        printf("Polygon is to large, not printing points.\n");
    }
    else
    {
        printf("Points:\t\t\tHilbert points:\n");
        for (int i = 0; i < size; i++)
        {
            points[i].print(false);
            printf("\t");
            hilbertPoints[i].print();
        }
    }
}

__host__ __device__ void GPUPolygon::printMatrix()
{
    const char *uncertainSymbol = "?";
    const char *emptySymbol = "\u00B7";
    const char *partialSymbol = "\U000025A0";
    const char *fullSymbol = "\U000025A3";

    printf("Rasterization matrix (%dx%d):\n", mbrWidth, mbrHeight);
    printf(" '%s': Uncertain \t'%s': Empty \t'%s': Partial \t'%s': Full\n",
           uncertainSymbol, emptySymbol, partialSymbol, fullSymbol);
    printf("   ");
    for (int i = 0; i < mbrWidth; i++)
    {
        printf("%d ", i % 10);
    }
    printf("\n");
    for (int y = 0; y < mbrHeight; y++)
    {
        printf("%2d ", y);
        for (int x = 0; x < mbrWidth; x++)
        {
            if (this->getMatrixXY(x, y) == UNCERTAIN_COLOR)
            {
                printf("%s ", uncertainSymbol);
            }
            else if (this->getMatrixXY(x, y) == EMPTY_COLOR)
            {
                printf("%s ", emptySymbol);
                // printf(" ");
            }
            else if (this->getMatrixXY(x, y) == PARTIAL_COLOR)
            {
                printf("%s ", partialSymbol);
                // printf("\U000025CF ");
            }
            else if (this->getMatrixXY(x, y) == FULL_COLOR)
            {
                printf("%s ", fullSymbol);
                // printf("\U000025EF ");
            }
            else
            {
                printf("%d ", this->getMatrixXY(x, y));
            }
        }
        printf("\n");
    }
}

__host__ __device__ GPUStack::GPUStack()
{
    size = 0;
    lastItem = NULL;
}

__host__ __device__ void GPUStack::push(int x, int y)
{
    GPUStackItem *item = new GPUStackItem;
    item->point = GPUPoint(x, y);

    item->prevItem = lastItem;
    lastItem = item;
    size++;
}

__host__ __device__ GPUPoint GPUStack::pop()
{
    // Stop execution if trying to pop from empty stack
    assert(size > 0);

    // Copy poped item
    GPUStackItem poped;
    poped.point = GPUPoint(lastItem->point);
    poped.prevItem = lastItem->prevItem;

    // Delete poped item
    lastItem->point.~GPUPoint();
    delete lastItem;

    // Change stack pointer & size
    lastItem = poped.prevItem;
    size--;

    return poped.point;
}

__host__ __device__ bool GPUStack::hasItems()
{
    return size > 0;
}

__host__ __device__ GPUStack::~GPUStack()
{
    while (size > 0)
    {
        this->pop();
    }
}

__host__ __device__ void GPUStack::print()
{
    GPUStackItem *currItem = lastItem;

    printf("Current stack items:\n");
    for (int i = size; i > 0; i--)
    {
        printf("%2d: ", i);
        currItem->point.print();
        currItem = currItem->prevItem;
    }
}