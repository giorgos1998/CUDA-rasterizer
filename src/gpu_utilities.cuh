/**
 * @file gpu_utilities.cuh
 * @brief This file contains __device__ functions that are used only by the GPU
 * kernels.
 */

#ifndef GPU_UTILITIES_H
#define GPU_UTILITIES_H

#include "gpu_containers.cuh"

/**
 * @brief Checks if test point is within the limit on y axis.
 * @param testPointY The y coordinate of the test point.
 * @param endCellY The y coordinate of the limit.
 * @param stepY The step on the y axis.
 * @returns True if the point is within the limit, false otherwise.
 */
__device__ bool checkYLimit(double testPointY, int endCellY, int stepY);

/**
 * @brief Performs a point-in-polygon test based on the pnpoly algorithm.
 * @param poly The target polygon.
 * @param testPoint The target point.
 * @returns True if the point is inside the polygon, false otherwise.
 */
__device__ bool isPointInsidePolygon(GPUPolygon &poly, GPUPoint testPoint);

/**
 * @brief Performs flood-fill in specified sector using the specified color.
 * @param poly The target polygon.
 * @param sectorMin The starting point of the sector (bottom-left).
 * @param sectorMax The ending point of the sector (top-right).
 * @param fillPoint The point from which the flood-fill starts.
 * @param fillColor The color to fill with.
 */
__device__ void floodFillSector(
    GPUPolygon &poly, GPUPoint sectorMin, GPUPoint sectorMax,
    GPUPoint fillPoint, int fillColor);

#endif