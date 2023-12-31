#include <stdio.h>
#include <float.h>
#include <vector>
#include <iostream>

// CPU timing library
#include <chrono>

#include "gpu_containers.cuh"
#include "kernels.cuh"
#include "host_utilities.cuh"

/**
 * @brief Rasterizes given polygon using the GPU.
 * 
 * @param poly The polygon to rasterize.
 * @param results An array to store the rasterization timings.
 * @param fillMethod Changes the fill algorithm to be used.
 * 1: Flood fill, 2: Per cell fill, 3: Hybrid algorithm based on polygon features.
 * (flood fill per sector or point-in-polygon checks per pixel)
 * @param printMatrix (Optional) Set to true to print the rasterization matrix.
 */
void CUDARasterize(
    GPUPolygon &poly, double* results, int fillMethod, bool printMatrix = false)
{
    // Cuda events for timing.
    cudaEvent_t memoryStart, memoryStop, prepStart, prepStop, borderStart,
        borderStop, fillStart, fillStop, outputStart, outputStop, totalStart,
        totalStop;
    float memoryMs, prepMs, borderMs, fillMs, outputMs, totalMs;

    // Pointers used by the device.
    GPUPolygon *poly_D;
    GPUPoint *points_D, *Hpoints_D;
    int *matrix_D;

    int matrixSize = poly.mbrHeight * poly.mbrWidth;
    int matrixThreads = (matrixSize > 1024) ? 1024 : matrixSize;
    int edgeThreads = (poly.size > 1024) ? 1024 : poly.size;

    // Initialize timing events.
    cudaEventCreate(&memoryStart);
    cudaEventCreate(&memoryStop);
    cudaEventCreate(&prepStart);
    cudaEventCreate(&prepStop);
    cudaEventCreate(&borderStart);
    cudaEventCreate(&borderStop);
    cudaEventCreate(&fillStart);
    cudaEventCreate(&fillStop);
    cudaEventCreate(&outputStart);
    cudaEventCreate(&outputStop);
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);

    // printf("In rasterizer\n");

    cudaEventRecord(totalStart);

    // Copy polygon to device.
    cudaEventRecord(memoryStart);
    cudaMalloc((void **)&poly_D, sizeof(GPUPolygon));
    cudaMemcpy(poly_D, &poly, sizeof(GPUPolygon), cudaMemcpyHostToDevice);
    // printf("Polygon to device\n");

    // Copy points to device.
    cudaMalloc((void **)&points_D, poly.size * sizeof(GPUPoint));
    cudaMalloc((void **)&Hpoints_D, poly.size * sizeof(GPUPoint));
    // cudaMemcpy(points_D, poly.points, poly.size * sizeof(GPUPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(
        Hpoints_D,
        poly.hilbertPoints,
        poly.size * sizeof(GPUPoint),
        cudaMemcpyHostToDevice);
    // printf("Points to device\n");

    // Copy polygon rasterization matrix to device.
    size_t mSize = matrixSize * sizeof(int);
    cudaMalloc((void **)&matrix_D, mSize);
    cudaMemcpy(matrix_D, poly.matrix, mSize, cudaMemcpyHostToDevice);
    // printf("Matrix to device\n");

    // Set device polygon's pointers to copied points & matrix.
    cudaMemcpy(
        &(poly_D->points),
        &points_D,
        sizeof(GPUPoint *),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        &(poly_D->hilbertPoints),
        &Hpoints_D,
        sizeof(GPUPoint *),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        &(poly_D->matrix),
        &matrix_D,
        sizeof(int *),
        cudaMemcpyHostToDevice);
    cudaEventRecord(memoryStop);

    // printf("Done with memory\n");

    // Prepare polygon points & rasterization matrix.
    cudaEventRecord(prepStart);
    normalizePoints<<<1, edgeThreads>>>(*poly_D);
    preparePolygonMatrix<<<1, matrixThreads>>>(*poly_D, matrixSize);
    cudaEventRecord(prepStop);
    // printf("Prepared matrix\n");

    // Rasterize border.
    cudaEventRecord(borderStart);
    edgeThreads = (poly.size - 1 > 1024) ? 1024 : poly.size - 1;
    rasterizeBorder<<<1, edgeThreads>>>(*poly_D);
    cudaEventRecord(borderStop);
    // printf("Rasterized border\n");

    cudaEventRecord(fillStart);
    if (fillMethod == 1)
    {
        // Use maximum amount of sectors.
        int xSectors = poly.mbrWidth;
        int ySectors = poly.mbrHeight;

        // Keep sector size between 4-5 cells.
        // int xSectors = ceilf(poly.mbrWidth / 5.0f);
        // int ySectors = ceilf(poly.mbrHeight / 5.0f);
        
        // Max number of sectors is 32*32.
        if (xSectors > 32) { xSectors = 32; }
        if (ySectors > 32) { ySectors = 32; }
        results[6] = xSectors * ySectors;
        // results[6] = 25;
        
        // Rasterize polygon using flood fill per sector.
        floodFillPolygonInSectors<<<1, xSectors * ySectors>>>(
            *poly_D, xSectors, ySectors);
        // floodFillPolygonInSectors<<<1, 25>>>(*poly_D, 5, 5);
    }
    else if (fillMethod == 2)
    {
        // Rasterize polygon using per pixel checks.
        fillPolygonPerPixel<<<1, matrixThreads>>>(*poly_D, matrixSize);
    }
    else if (fillMethod == 3)
    {
        float meanMBR = (poly.mbrWidth + poly.mbrHeight) / 2.0f;
        double expression;

        // Different exponential functions for each region.
        if (poly.size < 34) {
            expression = 623487.0199105 * exp(-0.173287 * poly.size);
        }
        else if (poly.size >= 34 && poly.size <= 41) {
            expression = 26007.978835 * exp(-0.0770164 * poly.size);
        }
        else {
            expression = 3010.0162402 * exp(-0.0256721 * poly.size);
        }

        if (expression < meanMBR) {
            results[7] = 1;
            // Use maximum amount of sectors.
            int xSectors = poly.mbrWidth;
            int ySectors = poly.mbrHeight;

            // Keep sector size between 4-5 cells.
            // int xSectors = ceilf(poly.mbrWidth / 5.0f);
            // int ySectors = ceilf(poly.mbrHeight / 5.0f);

            // Max number of sectors is 32*32.
            if (xSectors > 32) { xSectors = 32; }
            if (ySectors > 32) { ySectors = 32; }
            results[6] = xSectors * ySectors;
            // results[6] = 25;

            floodFillPolygonInSectors<<<1, xSectors * ySectors>>>(
                *poly_D, xSectors, ySectors);
            // floodFillPolygonInSectors<<<1, 25>>>(*poly_D, 5, 5);
        }
        else {
            results[7] = 2;
            fillPolygonPerPixel<<<1, matrixThreads>>>(*poly_D, matrixSize);
        }
        
    }
    cudaEventRecord(fillStop);
    // printf("Filled polygon\n");

    if (printMatrix) {
        printPolygon<<<1, 1>>>(*poly_D);
    }

    // Wait for GPU to finish all the kernels.
    cudaDeviceSynchronize();

    // Write back rasterization matrix to RAM.
    cudaEventRecord(outputStart);
    cudaMemcpy(poly.matrix, matrix_D, mSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(outputStop);

    cudaEventRecord(totalStop);

    // Free GPU memory used.
    cudaFree(poly_D);
    cudaFree(points_D);
    cudaFree(Hpoints_D);
    cudaFree(matrix_D);

    // Wait for the last event to complete (just in case)
    cudaEventSynchronize(totalStop);

    cudaEventElapsedTime(&memoryMs, memoryStart, memoryStop);
    cudaEventElapsedTime(&prepMs, prepStart, prepStop);
    cudaEventElapsedTime(&borderMs, borderStart, borderStop);
    cudaEventElapsedTime(&fillMs, fillStart, fillStop);
    cudaEventElapsedTime(&outputMs, outputStart, outputStop);
    cudaEventElapsedTime(&totalMs, totalStart, totalStop);

    results[0] = totalMs;
    results[1] = memoryMs;
    results[2] = prepMs;
    results[3] = borderMs;
    results[4] = fillMs;
    results[5] = outputMs;

    // printf("\nPolygon %d timings (%s fill):\n",
    //     poly.id, useFloodFill ? "flood" : "per cell" );
    // printf("Total: \t\t%f ms\n", totalMs);
    // printf("Memory: \t%f ms\n", memoryMs);
    // printf("Preperation: \t%f ms\n", prepMs);
    // printf("Border: \t%f ms\n", borderMs);
    // printf("Fill: \t\t%f ms\n", fillMs);
}

int main(void)
{
    std::vector<GPUPolygon> polygons;
    // int startLine = 1;      // Start from 1
    int setSize = 123045;   // Max dataset size: 1: 123045, 2: 2252316, 3: 3043
    int batch = 0;          // Start from 0
    int batchSize = 400000;
    int dataset = 1;

    int batches = ceil((float)setSize / (float)batchSize);
    int batchStart, batchEnd;
    // total, memory, prep, border, fill, output, sector size, used fill (hybrid)
    double runResults[8];
    timeMetrics avgResults;
    timeMetrics minResults;
    timeMetrics maxResults;
    double datasetMetrics[4] = { 0, 0, 0, 0 };
    // multiresultPoly graphResults[setSize - startLine + 1];

    // Prepare min and max result structs
    initResultStructs(avgResults, minResults, maxResults);

    // Create a test polygon
    // GPUPolygon testPoly = createTestPoly();
    // testPoly.print();
    // testPoly.printMatrix();

    // CUDARasterize(testPoly, runResults, true);
    // testPoly.print();
    // testPoly.printMatrix();
    // return 0;

    auto start = std::chrono::high_resolution_clock::now();

    batchStart = (batch * batchSize) + 1;
    batchEnd = (batch < batches - 1) ? (batch + 1) * batchSize : setSize;

    printf("=========== Batch %d of %d ===========\n", batch + 1, batches);
    printf("Start: %d, End: %d\n", batchStart, batchEnd);
    // Load polygons from dataset.
    loadPolygonsFromCSV(batchStart, batchEnd, dataset, polygons);
    // polygons.push_back(testPoly);

    for (int i = 0; i <= batchEnd - batchStart; i++)
    {
        printf("\rRasterizing polygon %d of %d (ID: %d)",
            i + batchStart, batchEnd, polygons[i].id);

        calculateMBR(polygons[i]);
        // normalizePointsCPU(polygons[i]);
        polygons[i].matrix = new int[polygons[i].mbrWidth * polygons[i].mbrHeight];
        // graphResults[i].polyID = polygons[i].id;

        // Flood fill run
        CUDARasterize(polygons[i], runResults, 1);
        gatherResults(runResults, avgResults, minResults, maxResults, polygons[i].id, 1);
        // graphResults[i].floodTime = runResults[4];

        // Per cell check run
        CUDARasterize(polygons[i], runResults, 2);
        gatherResults(runResults, avgResults, minResults, maxResults, polygons[i].id, 2);
        // graphResults[i].perCellTime = runResults[4];
        
        CUDARasterize(polygons[i], runResults, 3);
        gatherResults(runResults, avgResults, minResults, maxResults, polygons[i].id, 3);

        // polygons[i].print();
        // polygons[i].printMatrix();

        // Save metrics for the dataset
        datasetMetrics[0] += polygons[i].size;
        datasetMetrics[1] += polygons[i].mbrWidth * polygons[i].mbrHeight;
        datasetMetrics[2] += runResults[6];
        // Add 1 if flood fill was used by hybrid fill.
        datasetMetrics[3] += (runResults[7] == 1) ? 1 : 0;
    }
    printf("\nBatch rasterized!\n\n");
    
    for (int i = 0; i <= batchEnd - batchStart; i++)
    {
        delete[] polygons[i].matrix;
    }
    polygons.clear();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\nTotal dataset time: " << duration.count() << " ms" << std::endl;
    
    int numOfPolys = (batchEnd - batchStart) + 1;
    printResults(avgResults, minResults, maxResults, datasetMetrics, numOfPolys);
    // writeGraphResults(graphResults, setSize - startLine + 1);

    // Write rasterization results to file.
    // writeResultsToCSV(polygons);

    return 0;
}