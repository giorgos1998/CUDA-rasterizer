#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <float.h>

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
 * @param useFloodFill Changes the fill algorithm to be used.
 * (flood fill per sector or point-in-polygon checks per pixel)
 * @param printMatrix (Optional) Set to true to print the rasterization matrix.
 */
void CUDARasterize(
    GPUPolygon &poly, double* results, bool useFloodFill, bool printMatrix = false
) {
    // Cuda events for timing.
    cudaEvent_t memoryStart, memoryStop, prepStart, prepStop, borderStart,
        borderStop, fillStart, fillStop, totalStart, totalStop;
    float memoryMs, prepMs, borderMs, fillMs, totalMs;

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
    cudaMemcpy(Hpoints_D, poly.hilbertPoints, poly.size * sizeof(GPUPoint), cudaMemcpyHostToDevice);
    // printf("Points to device\n");

    // Copy polygon rasterization matrix to device.
    size_t mSize = matrixSize * sizeof(int);
    cudaMalloc((void **)&matrix_D, mSize);
    cudaMemcpy(matrix_D, poly.matrix, mSize, cudaMemcpyHostToDevice);
    // printf("Matrix to device\n");

    // Set device polygon's pointers to copied points & matrix.
    cudaMemcpy(&(poly_D->points), &points_D, sizeof(GPUPoint *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(poly_D->hilbertPoints), &Hpoints_D, sizeof(GPUPoint *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(poly_D->matrix), &matrix_D, sizeof(int *), cudaMemcpyHostToDevice);
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
    if (useFloodFill) {
        // Use maximum amount of sectors.
        int xSectors = poly.mbrWidth;
        int ySectors = poly.mbrHeight;

        // Keep sector size between 4-5 cells.
        // int xSectors = ceilf(poly.mbrWidth / 5.0f);
        // int ySectors = ceilf(poly.mbrHeight / 5.0f);
        
        // Max number of sectors is 32*32.
        if (xSectors > 32) { xSectors = 32; }
        if (ySectors > 32) { ySectors = 32; }
        
        // Rasterize polygon using flood fill per sector.
        floodFillPolygonInSectors<<<1, xSectors * ySectors>>>(*poly_D, xSectors, ySectors);
        // floodFillPolygonInSectors<<<1, 25>>>(*poly_D, 5, 5);
        cudaEventRecord(fillStop);
        results[5] = xSectors * ySectors;
        // results[5] = 25;
    } else {
        // Rasterize polygon using per pixel checks.
        fillPolygonPerPixel<<<1, matrixThreads>>>(*poly_D, matrixSize);
        cudaEventRecord(fillStop);
    }
    // printf("Filled polygon\n");

    cudaEventRecord(totalStop);

    if (printMatrix) {
        printPolygon<<<1, 1>>>(*poly_D);
    }

    // Wait for GPU to finish all the kernels.
    cudaDeviceSynchronize();

    // Write back rasterization matrix to RAM.
    cudaMemcpy(poly.matrix, matrix_D, mSize, cudaMemcpyDeviceToHost);

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
    cudaEventElapsedTime(&totalMs, totalStart, totalStop);

    results[0] = totalMs;
    results[1] = memoryMs;
    results[2] = prepMs;
    results[3] = borderMs;
    results[4] = fillMs;

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
    int startLine = 0;      // Start from 0
    int endLine = 123044;   // Max line: 123044
    double runResults[6];   // total, memory, prep, border, fill, sector size
    // total(flood), total(/cell), memory, prep, border, fill(flood), fill(/cell)
    double avgResults[7];
    double minResults[7];
    double maxResults[7];
    double datasetMetrics[3] = { 0, 0, 0 };

    // Prepare min and max result arrays
    for (int i = 0; i < 7; i++)
    {
        minResults[i] = DBL_MAX;
        maxResults[i] = DBL_MIN;
    }

    // Create a test polygon
    // GPUPolygon testPoly = createTestPoly();
    // testPoly.print();
    // testPoly.printMatrix();

    // CUDARasterize(testPoly, false, true, runResults);
    // testPoly.print();
    // testPoly.printMatrix();
    // return 0;

    printf("Loading dataset...\n");
    loadPolygonsFromCSV(startLine, endLine, polygons);
    printf("Dataset loaded!\n\n");
    // printf("In main: \t%p\n", &(polygons[0].hilbertPoints[0]));
    // for (int i = 0; i < polygons[0].size; i++)
    // {
    //     polygons[0].hilbertPoints[i].print();
    // }
    // polygons[0].hilbertPoints[polygons[0].size-1].print(); 

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i <= endLine - startLine; i++)
    {
        printf("\rRasterizing polygon %d of %d (ID: %d)",
            i, endLine - startLine, polygons[i].id);

        calculateMBR(polygons[i]);
        // normalizePointsCPU(polygons[i]);
        polygons[i].matrix = new int[polygons[i].mbrWidth * polygons[i].mbrHeight];
        // polygons[i].print();
        // polygons[i].printMatrix();

        // Flood fill run
        CUDARasterize(polygons[i], runResults, true);
        // Save average results
        avgResults[0] += runResults[0];
        avgResults[2] += runResults[1];
        avgResults[3] += runResults[2];
        avgResults[4] += runResults[3];
        avgResults[5] += runResults[4];
        // Save min results
        if (runResults[0] < minResults[0]) { minResults[0] = runResults[0]; }
        if (runResults[1] < minResults[2]) { minResults[2] = runResults[1]; }
        if (runResults[2] < minResults[3]) { minResults[3] = runResults[2]; }
        if (runResults[3] < minResults[4]) { minResults[4] = runResults[3]; }
        if (runResults[4] < minResults[5]) { minResults[5] = runResults[4]; }
        // Save max results
        if (runResults[0] > maxResults[0]) { maxResults[0] = runResults[0]; }
        if (runResults[1] > maxResults[2]) { maxResults[2] = runResults[1]; }
        if (runResults[2] > maxResults[3]) { maxResults[3] = runResults[2]; }
        if (runResults[3] > maxResults[4]) { maxResults[4] = runResults[3]; }
        if (runResults[4] > maxResults[5]) { maxResults[5] = runResults[4]; }

        // Per cell check run
        CUDARasterize(polygons[i], runResults, false);
        // Save average results
        avgResults[1] += runResults[0];
        avgResults[2] += runResults[1];
        avgResults[3] += runResults[2];
        avgResults[4] += runResults[3];
        avgResults[6] += runResults[4];
        // Save min results
        if (runResults[0] < minResults[1]) { minResults[1] = runResults[0]; }
        if (runResults[1] < minResults[2]) { minResults[2] = runResults[1]; }
        if (runResults[2] < minResults[3]) { minResults[3] = runResults[2]; }
        if (runResults[3] < minResults[4]) { minResults[4] = runResults[3]; }
        if (runResults[4] < minResults[6]) { minResults[6] = runResults[4]; }
        // Save max results
        if (runResults[0] > maxResults[1]) { maxResults[1] = runResults[0]; }
        if (runResults[1] > maxResults[2]) { maxResults[2] = runResults[1]; }
        if (runResults[2] > maxResults[3]) { maxResults[3] = runResults[2]; }
        if (runResults[3] > maxResults[4]) { maxResults[4] = runResults[3]; }
        if (runResults[4] > maxResults[6]) { maxResults[6] = runResults[4]; }

        // Save metrics for the dataset
        datasetMetrics[0] += polygons[i].size;
        datasetMetrics[1] += polygons[i].mbrWidth * polygons[i].mbrHeight;
        datasetMetrics[2] += runResults[5];
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\n\nTotal dataset time: " << duration.count() << " ms" << std::endl;
    
    int numOfPolys = (endLine - startLine) + 1;
    printf("\n -------------- Total results: -------------\n");
    printf(" Total time (flood fill): \t%10.3f ms\n",    avgResults[0]);
    printf(" Total time (per cell fill): \t%10.3f ms\n", avgResults[1]);
    printf(" Memory transfer time: \t\t%10.3f ms\n",     avgResults[2]);
    printf(" Preparation time: \t\t%10.3f ms\n",         avgResults[3]);
    printf(" Border rasterization time: \t%10.3f ms\n",  avgResults[4]);
    printf(" Fill time (flood fill): \t%10.3f ms\n",     avgResults[5]);
    printf(" Fill time (per cell fill): \t%10.3f ms\n",  avgResults[6]);
    printf("Note: memory, preparation & border times are for both flood fill \
and per cell runs\n\n");

    printf(" ------------- Average results: ------------\n");
    printf(" Total time (flood fill): \t%10.3f ms\n",    avgResults[0] / numOfPolys);
    printf(" Total time (per cell fill): \t%10.3f ms\n", avgResults[1] / numOfPolys);
    printf(" Memory transfer time: \t\t%10.3f ms\n",     avgResults[2] / (numOfPolys * 2));
    printf(" Preparation time: \t\t%10.3f ms\n",         avgResults[3] / (numOfPolys * 2));
    printf(" Border rasterization time: \t%10.3f ms\n",  avgResults[4] / (numOfPolys * 2));
    printf(" Fill time (flood fill): \t%10.3f ms\n",     avgResults[5] / numOfPolys);
    printf(" Fill time (per cell fill): \t%10.3f ms\n\n",avgResults[6] / numOfPolys);

    printf(" ------------- Minimum results: ------------\n");
    printf(" Total time (flood fill): \t%10.3f ms\n",       minResults[0]);
    printf(" Total time (per cell fill): \t%10.3f ms\n",    minResults[1]);
    printf(" Memory transfer time: \t\t%10.3f ms\n",        minResults[2]);
    printf(" Preparation time: \t\t%10.3f ms\n",            minResults[3]);
    printf(" Border rasterization time: \t%10.3f ms\n",     minResults[4]);
    printf(" Fill time (flood fill): \t%10.3f ms\n",        minResults[5]);
    printf(" Fill time (per cell fill): \t%10.3f ms\n\n",   minResults[6]);

    printf(" ------------- Maximum results: ------------\n");
    printf(" Total time (flood fill): \t%10.3f ms\n",       maxResults[0]);
    printf(" Total time (per cell fill): \t%10.3f ms\n",    maxResults[1]);
    printf(" Memory transfer time: \t\t%10.3f ms\n",        maxResults[2]);
    printf(" Preparation time: \t\t%10.3f ms\n",            maxResults[3]);
    printf(" Border rasterization time: \t%10.3f ms\n",     maxResults[4]);
    printf(" Fill time (flood fill): \t%10.3f ms\n",        maxResults[5]);
    printf(" Fill time (per cell fill): \t%10.3f ms\n\n",   maxResults[6]);

    float avgMBR = datasetMetrics[1] / numOfPolys;
    float avgSectors = datasetMetrics[2] / numOfPolys;
    printf(" ------------- Dataset metrics: ------------\n");
    printf(" Dataset size: \t\t\t%10d polygons\n",              numOfPolys);
    printf(" Average vetrices per polygon: \t%10.3f vertices\n",datasetMetrics[0] / numOfPolys);
    printf(" Average MBR per polygon: \t%10.3f cells\n",        avgMBR);
    printf(" Average sectors per polygon: \t%10.3f sectors\n",  avgSectors);
    printf(" Average sector size: \t\t%10.3f cells\n\n",        avgMBR / avgSectors);

    // polygons[0].matrix = new int[polygons[0].mbrWidth * polygons[0].mbrHeight];
    // polygons[0].print();
    // polygons[0].printMatrix();

    return 0;
}