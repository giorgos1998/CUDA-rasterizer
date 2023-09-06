/**
 * @file host_utilities.cuh
 * @brief This file contails __host__ functions that are used by the CPU.
 */

#ifndef HOST_UTILITIES_H
#define HOST_UTILITIES_H

#include "gpu_containers.cuh"

/** A struct that stores the Hilbert ID of a point and its rasterization value. */
struct hilbertID
{
    int id;
    int value;
};

/**
 * @brief Calculates given polygon's MBR (Minimum Bounding Rectangle).
 * @param poly The target polygon.
 */
__host__ void calculateMBR(GPUPolygon &poly);

/**
 * @brief Moves the polygon so that its MBR starts from (0,0).
 * @param poly The target polygon.
 */
__host__ void normalizePointsCPU(GPUPolygon &poly);

/**
 * @brief Loads polygons from ```MAPPED_CSV``` file.
 * @param startLine The first line to start loading polygons from.
 * @param endLine The last line to load a polygon from (inclusive).
 * @param polygons A list of polygons to load them to.
 */
__host__ void loadPolygonsFromCSV(
    int startLine, int endLine, std::vector<GPUPolygon> &polygons);

/**
 * @brief Create a polygon for testing purposes.
 * @param normalize (Optional) Set true to normalize the polygon using the CPU.
 * @returns The created polygon.
 */
__host__ GPUPolygon createTestPoly(bool normalize = false);

/**
 * @brief Calculates a point's Hilbert ID based on its coordinates.
 * @param x The x coordinate.
 * @param y The y coordinate.
 * @returns The calculated Hilbert ID.
 */
__host__ int convertPointToHilbertID(int x, int y);

/**
 * @brief Writes rasterization results in ```OUTPUT_CSV``` file, overwriting
 * previous results.
 * @param polygons The list of polygons to write.
 */
__host__ void writeResultsToCSV(std::vector<GPUPolygon> &polygons);

#endif