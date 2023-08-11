#include <stdio.h>
#include <assert.h>
#include "gpu_containers.h"

// Creates a new (0,0) point.
__host__ __device__ GPUPoint::GPUPoint()
{
    this->x = 0;
    this->y = 0;
}

/**
 * @brief Creates a new point with given coordinates.
 *
 * @param x The X coordinate.
 * @param y The Y coordinate.
 */
__host__ __device__ GPUPoint::GPUPoint(float x, float y)
{
    this->x = x;
    this->y = y;
}

// Copy constructor
__host__ __device__ GPUPoint::GPUPoint(GPUPoint &that)
{
    this->x = that.x;
    this->y = that.y;
}

// Destructor
__host__ __device__ GPUPoint::~GPUPoint()
{
    // printf("Deleting point\n");
}

// Prints the coordinates of the point.
__host__ __device__ void GPUPoint::print()
{
    printf("(%f, %f)\n", x, y);
}

/**
 * @brief Creates a new Polygon that can be used from host and device.
 *
 * @param size The number of points (vertices) the polygon has.
 * @param points The array of points.
 */
__host__ __device__ GPUPolygon::GPUPolygon(int size, GPUPoint points[])
{
    this->size = size;
    this->hilbertPoints = points;
    this->points = new GPUPoint[size];
}

// Copy constructor
__host__ __device__ GPUPolygon::GPUPolygon(GPUPolygon &that)
{
    // printf("In copy constructor\n");

    this->size = that.size;
    this->points = new GPUPoint[this->size];

    for (int i = 0; i < size; i++)
    {
        this->points[i] = that.points[i];
    }
}

// Copy assignment operator
__host__ __device__ GPUPolygon &GPUPolygon::operator=(const GPUPolygon &that)
{
    // printf("In copy assignment operator\n");
    if (this != &that)
    {
        // Using assert to work both on host and device.
        // Stop execution if the polygons don't have the same size.
        assert(this->size == that.size);

        this->size = that.size;
        this->points = new GPUPoint[this->size];

        for (int i = 0; i < size; i++)
        {
            this->points[i] = that.points[i];
        }
    }
    return *this;
}

// Destructor
__host__ __device__ GPUPolygon::~GPUPolygon()
{
    // printf("Deleting polygon\n");
}

// Prints the polygon points.
__host__ __device__ void GPUPolygon::print()
{
    printf("----- Polygon -----\n");
    printf("Hilbert min: ");
    hMin.print();
    printf("Hilbert max: ");
    hMax.print();
    printf("Points:\n");
    for (int i = 0; i < size; i++)
    {
        points[i].print();
    }
}

// Prints polygon's rasterization matrix.
__host__ __device__ void GPUPolygon::printMatrix()
{
    printf("Rasterization matrix:\n");
    printf("Size (WxH): %dx%d\n", mbrWidth, mbrHeight);
    printf("   ");
    for (int i = 0; i < mbrWidth; i++)
    {
        printf("%d ", i%10);
    }
    printf("\n");
    for (int i = 0; i < mbrHeight; i++)
    {
        printf("%2d ", i);
        for (int j = 0; j < mbrWidth; j++)
        {
            if (matrix[(i * mbrWidth) + j] == 3)
            {
                printf("? ");
            }
            else if (matrix[(i * mbrWidth) + j] == 0)
            {
                // printf(" ");
                printf("\u00B7 ");
            }
            else if (matrix[(i * mbrWidth) + j] == 1)
            {
                printf("\U000025A0 ");
                // printf("\U000025CF ");
            }
            else if (matrix[(i * mbrWidth) + j] == 2)
            {
                printf("\U000025A3 ");
            }
            else
            {
                printf("%d ", matrix[(i * mbrWidth) + j]);
            }
        }
        printf("\n");
    }
}

// Creates a new empty stack
__host__ __device__ GPUStack::GPUStack()
{
    size = 0;
    lastItem = NULL;
}

/**
 * @brief Add an item in the stack.
 * 
 * @param x The X coordinate of the point to add.
 * @param y The Y coordinate of the point to add.
 */
__host__ __device__ void GPUStack::push(int x, int y)
{
    GPUStackItem *item = new GPUStackItem;
    item->point = GPUPoint(x, y);

    item->prevItem = lastItem;
    lastItem = item;
    size++;

    // printf("Added ");
    // item->point.print();
    // printf("Current item %p\n", item);
    // printf("Previous item %p\n", item->prevItem);
}

// Removes and returns last item from the stack.
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

    // printf("Removed ");
    // poped.point.print();

    return poped.point;
}

// Destructor
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