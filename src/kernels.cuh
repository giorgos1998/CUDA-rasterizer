/**
 * @file kernels.cuh
 * @brief This file contails __global__ functions that are used to initiate
 * a new kernel at the GPU.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "gpu_containers.cuh"

/**
 * @brief Prints polygon info and rasterization matrix from the GPU. Best to
 * use with a single thread.
 * @param poly The target polygon.
 */
__global__ void printPolygon(GPUPolygon &poly);

/**
 * @brief Moves the polygon so that its MBR starts from (0,0).
 * Better use ```polygon.size``` threads if possible, but any number is safe.
 * @param poly The target polygon.
 */
__global__ void normalizePoints(GPUPolygon &poly);

/**
 * @brief Fills all cells of the rasterization matrix with UNCERTAIN_COLOR.
 * Better use ```mSize``` threads if possible, but any number is safe.
 * @param poly The target polygon.
 * @param mSize The size of the rasterization matrix (Width x Height).
 */
__global__ void preparePolygonMatrix(GPUPolygon &poly, int mSize);

/**
 * @brief Rasterizes given polygon's border (partial cells).
 * Better use ```polygon.size - 1``` threads if possible, but any number is safe.
 * @param poly The target polygon.
 */
__global__ void rasterizeBorder(GPUPolygon &poly);

/**
 * @brief Performs point-in-polygon checks to fill the rasterization matrix AFTER
 * the border rasterization. Better use ```matrixSize``` threads if possible,
 * but any number is safe.
 * @param poly The target polygon.
 * @param matrixSize The size of the rasterization matrix (Width x Height).
 */
__global__ void fillPolygonPerPixel(GPUPolygon &poly, int matrixSize);

/**
 * @brief Splits the rasterization matrix in smaller sectors and performs flood
 * fill to each one. The number of sectors is ```xSectors * ySectors```.
 * MUST use threads equal to the number of sectors.
 * @param poly The target polygon.
 * @param xSectors The number of sectors at the x axis.
 * @param ySectors The number of sectors at the y axis.
 */
__global__ void floodFillPolygonInSectors(
    GPUPolygon &poly, int xSectors, int ySectors);

#endif