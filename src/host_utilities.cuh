/**
 * @file host_utilities.cuh
 * @brief This file contails __host__ functions that are used by the CPU.
 */

#ifndef HOST_UTILITIES_H
#define HOST_UTILITIES_H

#include "gpu_containers.cuh"

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
 * @param dataset The TxNA dataset to choose, where x is the number.
 * @param polygons A list of polygons to load them to.
 */
__host__ void loadPolygonsFromCSV(
    int startLine, int endLine, int dataset, std::vector<GPUPolygon> &polygons);

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
__host__ uint convertPointToHilbertID(int x, int y);

/**
 * @brief Writes rasterization results in ```OUTPUT_CSV``` file, overwriting
 * previous results.
 * @param polygons The list of polygons to write.
 */
__host__ void writeResultsToCSV(std::vector<GPUPolygon> &polygons);

/**
 * @brief Stores run results in given arrays to use them later.
 * @param run The run results array.
 * @param avg The averages array. The average is not calculated yet, it just
 * stores the sum of the results.
 * @param min The minimums array.
 * @param max The maximums array.
 * @param polyID The ID of the current polygon.
 * @param fillMethod The fill method used.
 * 1: Flood fill, 2: Per cell fill, 3: Hybrid algorithm.
 */
__host__ void gatherResults(
    double *run, timeMetrics &avg, timeMetrics &min, timeMetrics &max,
    int polyID, int fillMethod);

/**
 * @brief Prints the total results of the runs.
 * @param avg The averages array. The average is not calculated yet, it just
 * stores the sum of the results.
 * @param min The minimums array.
 * @param max The maximums array.
 * @param dataset The dataset metrics array.
 * @param numOfPolys The number of polygons in dataset.
 */
__host__ void printResults(
    timeMetrics &avg, timeMetrics &min, timeMetrics &max, double *dataset,
    int numOfPolys);

/**
 * @brief Writes timing results in plot-able format in ```GRAPH_CSV``` file,
 * overwriting previous results.
 * @param results The plot-able results array.
 * @param size The size of the array.
 */
__host__ void writeGraphResults(multiresultPoly *results, int size);

/**
 * @brief Fills structs with the correct initial values.
 * @param avg The averages struct.
 * @param min The minimums struct.
 * @param max The maximums struct.
 */
__host__ void initResultStructs(
    timeMetrics &avg, timeMetrics &min, timeMetrics &max);

#endif