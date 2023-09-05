/**
 * @file host_utilities.cuh
 * @brief This file contails __host__ functions that are used by the CPU.
 */

#ifndef HOST_UTILITIES_H
#define HOST_UTILITIES_H

// Libraries for file reading
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "gpu_containers.cuh"

// Dataset file of polygons mapped to Hilbert space
#define MAPPED_CSV "T1NA_mapped.csv"

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

#endif