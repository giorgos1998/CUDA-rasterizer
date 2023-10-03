/**
 * @file host_utilities.cu
 * @brief This file contails __host__ functions that are used by the CPU.
 */

// Libraries for file reading
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
// Library for double min/max
#include <float.h>

#include "host_utilities.cuh"
#include "gpu_containers.cuh"
#include "constants.h"

__host__ void calculateMBR(GPUPolygon &poly)
{
    poly.hMax = GPUPoint(poly.hilbertPoints[0].x, poly.hilbertPoints[0].y);
    poly.hMin = GPUPoint(poly.hilbertPoints[0].x, poly.hilbertPoints[0].y);

    for (int i = 1; i < poly.size - 1; i++)
    {
        if (poly.hilbertPoints[i].x < poly.hMin.x)
        {
            poly.hMin.x = poly.hilbertPoints[i].x;
        }
        if (poly.hilbertPoints[i].y < poly.hMin.y)
        {
            poly.hMin.y = poly.hilbertPoints[i].y;
        }
        if (poly.hilbertPoints[i].x > poly.hMax.x)
        {
            poly.hMax.x = poly.hilbertPoints[i].x;
        }
        if (poly.hilbertPoints[i].y > poly.hMax.y)
        {
            poly.hMax.y = poly.hilbertPoints[i].y;
        }
    }

    // Round MBR and add 1 cell buffer around
    poly.hMin.x = (int)poly.hMin.x - 1;
    poly.hMin.y = (int)poly.hMin.y - 1;
    poly.hMax.x = (int)poly.hMax.x + 1;
    poly.hMax.y = (int)poly.hMax.y + 1;

    poly.mbrWidth = poly.hMax.x - poly.hMin.x + 1;
    poly.mbrHeight = poly.hMax.y - poly.hMin.y + 1;
}

__host__ void normalizePointsCPU(GPUPolygon &poly)
{
    for (int i = 0; i < poly.size; i++)
    {
        poly.points[i] = GPUPoint(
            poly.hilbertPoints[i].x - poly.hMin.x,
            poly.hilbertPoints[i].y - poly.hMin.y);
    }
}

__host__ void loadPolygonsFromCSV(
    int startLine, int endLine, std::vector<GPUPolygon> &polygons)
{
    std::ifstream fin;
    std::string line, token, coordToken;
    std::vector<GPUPoint> points;
    int polyID;
    double x, y;

    printf("Loading dataset '%s'...\n", MAPPED_CSV);
    fin.open(MAPPED_CSV);

    if (!fin.good())
    {
        printf("ERROR: dataset file could not be opened.\n");
        return;
    }

    // Skip lines until the desired starting line.
    for (int i = 1; i < startLine; i++)
    {
        getline(fin, line);
    }

    // Read & parse lines with polygons.
    for (int i = startLine; i <= endLine; i++)
    {
        points.clear();

        // Get the whole line in a stream.
        getline(fin, line);
        std::stringstream lineStream(line);

        // Parse polygon ID at beginning of the line.
        getline(lineStream, token, ',');
        polyID = std::stoi(token);

        // Parse polygon points.
        while (getline(lineStream, token, ','))
        {
            std::stringstream tokenStream(token);

            getline(tokenStream, coordToken, ' ');
            x = std::stod(coordToken);

            getline(tokenStream, coordToken, ' ');
            y = std::stod(coordToken);

            // GPUPoint newPoint = GPUPoint(x, y);
            points.push_back(GPUPoint(x, y));
        }

        // Add parsed polygon to list.
        polygons.push_back(GPUPolygon(polyID, points.size(), &points[0]));
    }
    printf("Dataset loaded!\n\n");
}

__host__ GPUPolygon createTestPoly(bool normalize)
{
    GPUPoint testPoints[] = {
        GPUPoint(1.5, 1.5),
        GPUPoint(20.5, 45.5),
        GPUPoint(62.5, 62.5),
        GPUPoint(45.5, 20.5),
        GPUPoint(1.5, 1.5)};
    GPUPolygon poly = GPUPolygon(1, 5, testPoints);

    calculateMBR(poly);
    poly.matrix = new int[poly.mbrWidth * poly.mbrHeight];

    if (normalize)
    {
        normalizePointsCPU(poly);
    }

    return poly;
}

__host__ uint convertPointToHilbertID(int x, int y)
{
    // Function is copied by the serial code, no idea how it works.
    uint rx, ry, s;
    uint d = 0;

    for (s = HILBERT_SIZE / 2; s > 0; s /= 2)
    {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);

        // Rotate/flip the quadrant appropriately.
        if (ry == 0)
        {
            if (rx == 1)
            {
                x = s - 1 - x;
                y = s - 1 - y;
            }

            int temp = x;
            x = y;
            y = temp;
        }
    }
    return d;
}

__host__ void writeResultsToCSV(std::vector<GPUPolygon> &polygons)
{
    std::ofstream fout;
    std::vector<hilbertID> hilbertIDs;
    int pointValue, hilbertX, hilbertY;

    printf("Writing rasterization results to '%s'...\n", OUTPUT_CSV);
    fout.open(OUTPUT_CSV, std::ios::trunc);
    if (fout.is_open())
    {
        // Add header to file.
        fout << "Polygon ID,[Hilbert ID] [1: Partial / 2: Full]\n";

        for (int i = 0; i < polygons.size(); i++)
        {
            for (int y = 0; y < polygons[i].mbrHeight; y++)
            {
                for (int x = 0; x < polygons[i].mbrWidth; x++)
                {
                    pointValue = polygons[i].getMatrixXY(x, y);

                    if (pointValue == PARTIAL_COLOR || pointValue == FULL_COLOR)
                    {
                        // Move coordinates back to Hilbert space.
                        hilbertX = x + polygons[i].hMin.x;
                        hilbertY = y + polygons[i].hMin.y;

                        hilbertIDs.push_back(
                            {convertPointToHilbertID(hilbertX, hilbertY), pointValue});
                    }
                }
            }
            // Sort by Hilbert ID in ascending order.
            std::sort(
                hilbertIDs.begin(),
                hilbertIDs.end(),
                [](hilbertID a, hilbertID b) // sorting lambda function
                {
                    return a.id < b.id;
                });

            // Write polygon info to file.
            fout << polygons[i].id;
            for (int j = 0; j < hilbertIDs.size(); j++)
            {
                fout << "," << hilbertIDs[j].id << " " << hilbertIDs[j].value;
            }
            fout << "\n";

            hilbertIDs.clear();
        }
        fout.close();
        printf("Writing completed!\n");
    }
    else
    {
        printf("ERROR: output file could not be opened.\n");
    }
}

__host__ void gatherResults(
    double *run, timeMetrics &avg, timeMetrics &min, timeMetrics &max,
    int polyID, int fillMethod)
{
    // ---------------- Averages ----------------
    if (fillMethod == 1)
    {
        // Flood fill
        avg.floodTotal.time += run[0];
        avg.floodFill.time += run[4];
    }
    else if (fillMethod == 2)
    {
        // Per cell fill
        avg.perCellTotal.time += run[0];
        avg.perCellFill.time += run[4];
    }
    else
    {
        // Hybrid fill
        avg.hybridTotal.time += run[0];
        avg.hybridFill.time += run[4];
    }
    avg.memory.time += run[1];
    avg.preparation.time += run[2];
    avg.border.time += run[3];
    avg.output.time += run[5];

    // ---------------- Minimums ----------------
    if (fillMethod == 1)
    {
        // Flood fill
        if (run[0] < min.floodTotal.time)
        {
            min.floodTotal.time = run[0];
            min.floodTotal.polyID = polyID;
        }
        if (run[4] < min.floodFill.time)
        {
            min.floodFill.time = run[4];
            min.floodFill.polyID = polyID;
        }
    }
    else if (fillMethod == 2)
    {
        // Per cell fill
        if (run[0] < min.perCellTotal.time)
        {
            min.perCellTotal.time = run[0];
            min.perCellTotal.polyID = polyID;
        }
        if (run[4] < min.perCellFill.time)
        {
            min.perCellFill.time = run[4];
            min.perCellFill.polyID = polyID;
        }
    }
    else
    {
        // Hybrid fill
        if (run[0] < min.hybridTotal.time)
        {
            min.hybridTotal.time = run[0];
            min.hybridTotal.polyID = polyID;
        }
        if (run[4] < min.hybridFill.time)
        {
            min.hybridFill.time = run[4];
            min.hybridFill.polyID = polyID;
        }
    }

    if (run[1] < min.memory.time)
    {
        min.memory.time = run[1];
        min.memory.polyID = polyID;
    }
    if (run[2] < min.preparation.time)
    {
        min.preparation.time = run[2];
        min.preparation.polyID = polyID;
    }
    if (run[3] < min.border.time)
    {
        min.border.time = run[3];
        min.border.polyID = polyID;
    }
    if (run[5] < min.output.time)
    {
        min.output.time = run[5];
        min.output.polyID = polyID;
    }

    // ---------------- Maximums ----------------
    if (fillMethod == 1)
    {
        // Flood fill
        if (run[0] > max.floodTotal.time)
        {
            max.floodTotal.time = run[0];
            max.floodTotal.polyID = polyID;
        }
        if (run[4] > max.floodFill.time)
        {
            max.floodFill.time = run[4];
            max.floodFill.polyID = polyID;
        }
    }
    else if (fillMethod == 2)
    {
        // Per cell fill
        if (run[0] > max.perCellTotal.time)
        {
            max.perCellTotal.time = run[0];
            max.perCellTotal.polyID = polyID;
        }
        if (run[4] > max.perCellFill.time)
        {
            max.perCellFill.time = run[4];
            max.perCellFill.polyID = polyID;
        }
    }
    else
    {
        // Hybrid fill
        if (run[0] > max.hybridTotal.time)
        {
            max.hybridTotal.time = run[0];
            max.hybridTotal.polyID = polyID;
        }
        if (run[4] > max.hybridFill.time)
        {
            max.hybridFill.time = run[4];
            max.hybridFill.polyID = polyID;
        }
    }

    if (run[1] > max.memory.time)
    {
        max.memory.time = run[1];
        max.memory.polyID = polyID;
    }
    if (run[2] > max.preparation.time)
    {
        max.preparation.time = run[2];
        max.preparation.polyID = polyID;
    }
    if (run[3] > max.border.time)
    {
        max.border.time = run[3];
        max.border.polyID = polyID;
    }
    if (run[5] > max.output.time)
    {
        max.output.time = run[5];
        max.output.polyID = polyID;
    }
}

__host__ void printResults(
    timeMetrics &avg, timeMetrics &min, timeMetrics &max, double *dataset,
    int numOfPolys)
{
    printf("\n -------------- Total results: -------------\n");
    printf(" Total time (flood fill):      %11.3f ms\n", avg.floodTotal.time);
    printf(" Total time (per cell fill):   %11.3f ms\n", avg.perCellTotal.time);
    printf(" Total time (hybrid fill):     %11.3f ms\n", avg.hybridTotal.time);
    printf(" Data transfer time (to GPU):  %11.3f ms\n", avg.memory.time);
    printf(" Preparation time:             %11.3f ms\n", avg.preparation.time);
    printf(" Border rasterization time:    %11.3f ms\n", avg.border.time);
    printf(" Fill time (flood fill):       %11.3f ms\n", avg.floodFill.time);
    printf(" Fill time (per cell fill):    %11.3f ms\n", avg.perCellFill.time);
    printf(" Fill time (hybrid fill):      %11.3f ms\n", avg.hybridFill.time);
    printf(" Data transfer time (to RAM):  %11.3f ms\n", avg.output.time);
    printf("Note: data, preparation & border times are for all 3 runs\n\n");

    printf(" ------------- Average results: ------------\n");
    printf(" Total time (flood fill):      %11.3f ms\n", avg.floodTotal.time / numOfPolys);
    printf(" Total time (per cell fill):   %11.3f ms\n", avg.perCellTotal.time / numOfPolys);
    printf(" Total time (hybrid fill):     %11.3f ms\n", avg.hybridTotal.time / numOfPolys);
    printf(" Data transfer time (to GPU):  %11.3f ms\n", avg.memory.time / (numOfPolys * 3));
    printf(" Preparation time:             %11.3f ms\n", avg.preparation.time / (numOfPolys * 3));
    printf(" Border rasterization time:    %11.3f ms\n", avg.border.time / (numOfPolys * 3));
    printf(" Fill time (flood fill):       %11.3f ms\n", avg.floodFill.time / numOfPolys);
    printf(" Fill time (per cell fill):    %11.3f ms\n", avg.perCellFill.time / numOfPolys);
    printf(" Fill time (hybrid fill):      %11.3f ms\n", avg.hybridFill.time / numOfPolys);
    printf(" Data transfer time (to RAM):  %11.3f ms\n\n", avg.output.time / (numOfPolys * 3));

    printf(" ------------- Minimum results: ------------\n");
    printf(" Total time (flood fill):      %11.3f ms (ID: %d)\n", min.floodTotal.time, min.floodTotal.polyID);
    printf(" Total time (per cell fill):   %11.3f ms (ID: %d)\n", min.perCellTotal.time, min.perCellTotal.polyID);
    printf(" Total time (hybrid fill):     %11.3f ms (ID: %d)\n", min.hybridTotal.time, min.hybridTotal.polyID);
    printf(" Data transfer time (to GPU):  %11.3f ms (ID: %d)\n", min.memory.time, min.memory.polyID);
    printf(" Preparation time:             %11.3f ms (ID: %d)\n", min.preparation.time, min.preparation.polyID);
    printf(" Border rasterization time:    %11.3f ms (ID: %d)\n", min.border.time, min.border.polyID);
    printf(" Fill time (flood fill):       %11.3f ms (ID: %d)\n", min.floodFill.time, min.floodFill.polyID);
    printf(" Fill time (per cell fill):    %11.3f ms (ID: %d)\n", min.perCellFill.time, min.perCellFill.polyID);
    printf(" Fill time (hybrid fill):      %11.3f ms (ID: %d)\n", min.hybridFill.time, min.hybridFill.polyID);
    printf(" Data transfer time (to RAM):  %11.3f ms (ID: %d)\n\n", min.output.time, min.output.polyID);

    printf(" ------------- Maximum results: ------------\n");
    printf(" Total time (flood fill):      %11.3f ms (ID: %d)\n", max.floodTotal.time, max.floodTotal.polyID);
    printf(" Total time (per cell fill):   %11.3f ms (ID: %d)\n", max.perCellTotal.time, max.perCellTotal.polyID);
    printf(" Total time (hybrid fill):     %11.3f ms (ID: %d)\n", max.hybridTotal.time, max.hybridTotal.polyID);
    printf(" Data transfer time (to GPU):  %11.3f ms (ID: %d)\n", max.memory.time, max.memory.polyID);
    printf(" Preparation time:             %11.3f ms (ID: %d)\n", max.preparation.time, max.preparation.polyID);
    printf(" Border rasterization time:    %11.3f ms (ID: %d)\n", max.border.time, max.border.polyID);
    printf(" Fill time (flood fill):       %11.3f ms (ID: %d)\n", max.floodFill.time, max.floodFill.polyID);
    printf(" Fill time (per cell fill):    %11.3f ms (ID: %d)\n", max.perCellFill.time, max.perCellFill.polyID);
    printf(" Fill time (hybrid fill):      %11.3f ms (ID: %d)\n", max.hybridFill.time, max.hybridFill.polyID);
    printf(" Data transfer time (to RAM):  %11.3f ms (ID: %d)\n\n", max.output.time, max.output.polyID);

    float avgMBR = dataset[1] / numOfPolys;
    float avgSectors = dataset[2] / numOfPolys;
    printf(" ------------- Dataset metrics: ------------\n");
    printf(" Dataset size:                 %11d polygons\n", numOfPolys);
    printf(" Average vetrices per polygon: %11.3f vertices\n", dataset[0] / numOfPolys);
    printf(" Average MBR per polygon:      %11.3f cells\n", avgMBR);
    printf(" Average sectors per polygon:  %11.3f sectors\n", avgSectors);
    printf(" Average sector size:          %11.3f cells\n\n", avgMBR / avgSectors);
}

__host__ void writeGraphResults(multiresultPoly *results, int size)
{
    std::ofstream fout;

    printf("Writing graph results to '%s'...\n", GRAPH_CSV);
    fout.open(GRAPH_CSV, std::ios::trunc);
    if (fout.is_open())
    {
        // Add header to file.
        fout << "Polygon ID,Flood fill time (ms),Per cell fill time (ms)\n";

        for (int i = 0; i < size; i++)
        {
            fout << results[i].polyID << ","
                 << results[i].floodTime << ","
                 << results[i].perCellTime << "\n";
        }
        fout.close();
        printf("Writing completed!\n");
    }
    else
    {
        printf("ERROR: output file could not be opened.\n");
    }
}

__host__ void initResultStructs(timeMetrics &avg, timeMetrics &min, timeMetrics &max)
{
    avg.floodTotal.time = 0;
    avg.perCellTotal.time = 0;
    avg.hybridTotal.time = 0;
    avg.memory.time = 0;
    avg.preparation.time = 0;
    avg.border.time = 0;
    avg.floodFill.time = 0;
    avg.perCellFill.time = 0;
    avg.hybridFill.time = 0;
    avg.output.time = 0;

    min.floodTotal.time = DBL_MAX;
    min.perCellTotal.time = DBL_MAX;
    min.hybridTotal.time = DBL_MAX;
    min.memory.time = DBL_MAX;
    min.preparation.time = DBL_MAX;
    min.border.time = DBL_MAX;
    min.floodFill.time = DBL_MAX;
    min.perCellFill.time = DBL_MAX;
    min.hybridFill.time = DBL_MAX;
    min.output.time = DBL_MAX;

    max.floodTotal.time = DBL_MIN;
    max.perCellTotal.time = DBL_MIN;
    max.hybridTotal.time = DBL_MIN;
    max.memory.time = DBL_MIN;
    max.preparation.time = DBL_MIN;
    max.border.time = DBL_MIN;
    max.floodFill.time = DBL_MIN;
    max.perCellFill.time = DBL_MIN;
    max.hybridFill.time = DBL_MIN;
    max.output.time = DBL_MIN;

    min.floodTotal.polyID = -1;
    min.perCellTotal.polyID = -1;
    min.hybridTotal.polyID = -1;
    min.memory.polyID = -1;
    min.preparation.polyID = -1;
    min.border.polyID = -1;
    min.floodFill.polyID = -1;
    min.perCellFill.polyID = -1;
    min.hybridFill.polyID = -1;
    min.output.polyID = -1;

    max.floodTotal.polyID = -1;
    max.perCellTotal.polyID = -1;
    max.hybridTotal.polyID = -1;
    max.memory.polyID = -1;
    max.preparation.polyID = -1;
    max.border.polyID = -1;
    max.floodFill.polyID = -1;
    max.perCellFill.polyID = -1;
    max.hybridFill.polyID = -1;
    max.output.polyID = -1;
}