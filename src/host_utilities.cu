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
        GPUPoint(11.5, 21.5),
        GPUPoint(31.5, 31.5),
        GPUPoint(21.5, 11.5),
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

__host__ int convertPointToHilbertID(int x, int y)
{
    // Function is copied by the serial code, no idea how it works.
    int rx, ry, s;
    int d = 0;

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