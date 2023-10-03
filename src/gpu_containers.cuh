/**
 * @file gpu_containers.cuh
 * @brief This file contains classes and structs used across the CUDA rasterizer.
 */

#ifndef GPU_CONTAINERS_H
#define GPU_CONTAINERS_H


/** A struct that stores the Hilbert ID of a point and its rasterization value. */
struct hilbertID
{
    uint id;
    int value;
};

/** A struct that stores a timing value and with the Polygon ID it belongs to. */
struct resultPoly
{
    int polyID;
    double time;
};

/** A struct that stores the polygon ID with its fill timings. */
struct multiresultPoly
{
    int polyID;
    double floodTime;
    double perCellTime;
};

/** A struct that stores the timing results of the rasterization. */
struct timeMetrics
{
    resultPoly floodTotal;
    resultPoly perCellTotal;
    resultPoly hybridTotal;
    resultPoly memory;
    resultPoly preparation;
    resultPoly border;
    resultPoly floodFill;
    resultPoly perCellFill;
    resultPoly hybridFill;
    resultPoly output;
};

/** A class used to create points with (x,y) coordinates. */
class GPUPoint
{
public:
    /** The x coordinate of the point. */
    double x;
    /** The y coordinate of the point. */
    double y;

    /**
     * @brief Creates a new point with given coordinates.
     * @param x The x coordinate.
     * @param y The y coordinate.
     */
    __host__ __device__ GPUPoint(double x, double y);

    /** @brief Creates a new (0,0) point. */
    __host__ __device__ GPUPoint();

    /**
     * @brief Creates a new point using another point's coordinates.
     * @param that The point to copy.
     */
    __host__ __device__ GPUPoint(const GPUPoint &that);

    /** Destroys current point. */
    __host__ __device__ ~GPUPoint();

    /**
     * @brief Checks that both of the points' coordinates are equal.
     * @param that The point to compare with.
     * @returns True if the points are equal, false otherwise.
     */
    __host__ __device__ bool operator==(const GPUPoint &that);

    /**
     * @brief Prints the point's coordinates.
     * @param newLine (Optional) Set false to remove '\\n' after the coordinates.
     */
    __host__ __device__ void print(bool newLine = true);
};

/** A class used to create polygons and store their info. */
class GPUPolygon
{
public:
    /** Moved points after normalization. 1st point is the same as the last one. */
    GPUPoint *points;
    /** Points in Hilbert space. 1st point is the same as the last one. */
    GPUPoint *hilbertPoints;
    /** The number of points the polygon has PLUS the repeated one. */
    int size;
    /** Polygon rasterization matrix (normalized coordinates). */
    int *matrix;
    /** The starting point (bottom-left) of polygon's MBR in Hilbert space. */
    GPUPoint hMin;
    /** The ending point (top-right) of polygon's MBR in Hilbert space. */
    GPUPoint hMax;
    /** The width of the polygon's MBR (and its rasterization matrix). */
    int mbrWidth;
    /** The height of the polygon's MBR (and its rasterization matrix). */
    int mbrHeight;
    /** The ID number of the polygon from its dataset. */
    int id;

    /**
     * @brief Creates a new polygon with given properties.
     * @param id The ID of the polygon (usually from its dataset).
     * @param size The number of points the polygon has PLUS the repeated one.
     * @param points The array of the polygons points in Hilbert space. The 1st
     * point should be repeated as the last one.
     */
    __host__ __device__ GPUPolygon(int id, int size, GPUPoint points[]);

    /**
     * @brief Creates a new polygon by deep-copying another polygon.
     * @param that The polygon to copy.
     */
    __host__ __device__ GPUPolygon(const GPUPolygon &that);

    /**
     * @brief Creates a new polygon by deep-copying another polygon.
     * @param that The polygon to copy.
     */
    __host__ __device__ GPUPolygon &operator=(const GPUPolygon &that);

    /** Destroys current polygon. */
    __host__ __device__ ~GPUPolygon();

    /**
     * @brief Get the value of the rasterization matrix in given coordinates.
     * @param x The x coordinate.
     * @param y The y coordinate.
     */
    __host__ __device__ int getMatrixXY(int x, int y);

    /**
     * @brief Set the value of the rasterization matrix in given coordinates.
     * @param x The x coordinate.
     * @param y The y coordinate.
     * @param value The value to set.
     */
    __host__ __device__ void setMatrixXY(int x, int y, int value);

    /** Prints polygon's info & points. */
    __host__ __device__ void print();

    /** Prints polygon's rasterization matrix. */
    __host__ __device__ void printMatrix();
};

/** A struct used to store a point in a GPUStack. */
struct GPUStackItem
{
    /** The stored point. */
    GPUPoint point;
    /** A pointer to the previous point in the stack. */
    GPUStackItem *prevItem;
};

/** A class used to create stacks of points. */
class GPUStack
{
public:
    /** The number of points in the stack. */
    int size;
    /** A pointer to the last point stored in the stack. */
    GPUStackItem *lastItem;

    /** Creates a new empty stack. */
    __host__ __device__ GPUStack();

    /** Destroys current stack and all its contents. */
    __host__ __device__ ~GPUStack();

    /**
     * @brief Removes the last point from the stack.
     * @returns The removed point.
     */
    __host__ __device__ GPUPoint pop();

    /**
     * @brief Checks if the stack has stored points.
     * @returns True if there are stored points in the stack, false otherwise.
     */
    __host__ __device__ bool hasItems();

    /**
     * @brief Adds a point at the top of the stack.
     * @param x The x coordinate of the point.
     * @param y The y coordinate of the point.
     */
    __host__ __device__ void push(int x, int y);

    /** Prints points currently in the stack. */
    __host__ __device__ void print();
};

#endif