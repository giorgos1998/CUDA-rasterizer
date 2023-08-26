#include <stdio.h>
#include <math.h>
#include <assert.h>
// Libraries for file reading
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "gpu_containers.h"

/**
 *  GLOBAL CONSTANTS
 */
// Rasterization cell marks
#define EMPTY_COLOR 0
#define PARTIAL_COLOR 1
#define FULL_COLOR 2
#define UNCERTAIN_COLOR 3
#define FULL_CHECKED 4
// Dataset file of polygons mapped to Hilbert space
#define MAPPED_CSV "T1NA_mapped.csv"

// Sample min/max
const GPUPoint S_MAX = GPUPoint(-66.8854, 49.3844);
const GPUPoint S_MIN = GPUPoint(-124.849, 24.5214);
// Hilbert space mix/max
const GPUPoint H_MAX = GPUPoint(65535, 65535); // 65535 = 2^16 - 1
const GPUPoint H_MIN = GPUPoint(0, 0);

__global__ void printPolygon(GPUPolygon &poly)
{
    // printf("Hello from the GPU!\n");
    // poly.print();
    poly.printMatrix();
}

__global__ void preparePolygonMatrix(GPUPolygon &poly, int mSize)
{
    // ID of current thread in reference to all created threads in kernel
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The number of threads in kernel
    int stride = blockDim.x * gridDim.x;

    // Create polygon rasterization matrix using a grid-stride loop
    for (int i = index; i < mSize; i += stride)
    {
        // poly.matrix[i] = i + 1;
        poly.matrix[i] = UNCERTAIN_COLOR;
    }
}

__device__ bool checkYLimit(double testPointY, int endCellY, int stepY)
{
    if (stepY > 0) {
        return (testPointY < endCellY + stepY);
    } else {
        return (testPointY > endCellY);
    }
}

__global__ void rasterizeBorder(GPUPolygon &poly)
{
    GPUPoint startPoint, endPoint;
    GPUPoint startCell, endCell;
    GPUPoint step;
    int nextVertical, nextHorizontal;
    double gradient, edgeLength, intersectY, intersectX, distance;
    GPUPoint tMax, tDelta;
    // int checkID = 23;

    // ID of current thread in reference to all created threads in kernel.
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;

    // Number of threads MUST be equal to the edges of the polygon.
    assert(t_ID < poly.size);

    // Find edge stard/end with positive orientation (on x axis).
    if (poly.points[t_ID].x < poly.points[t_ID + 1].x)
    {
        startPoint = GPUPoint(poly.points[t_ID]);
        endPoint = GPUPoint(poly.points[t_ID + 1]);
    }
    else
    {
        startPoint = GPUPoint(poly.points[t_ID + 1]);
        endPoint = GPUPoint(poly.points[t_ID]);
    }
    // printf("Edge %d\n", t_ID + 1);
    // printf("Starting point: ");
    // startPoint.print();
    // printf("Ending point: ");
    // endPoint.print();

    startCell.x = (int)startPoint.x;
    startCell.y = (int)startPoint.y;
    endCell.x = (int)endPoint.x;
    endCell.y = (int)endPoint.y;

    // Check if edge is contained in only one cell.
    if (startCell == endCell)
    {
        poly.setMatrixXY(startCell.x, startCell.y, PARTIAL_COLOR);
        // printf("Thread %d stopped early at: (%d, %d)\n", t_ID, (int)startCell.x, (int)startCell.y);
        return;
    }
    // else
    // {
    //     printf("Thread %d continued\n", t_ID);
    // }
    
    // if (t_ID == checkID) {
    //     printf("=============== Thread %d ==================\n", checkID);
    //     printf("Starting point: ");
    //     startPoint.print();
    //     printf("Ending point: ");
    //     endPoint.print();
    //     printf("Starting cell: ");
    //     startCell.print();
    //     printf("Ending cell: ");
    //     endCell.print();
    // }

    // Edge always goes from smaller X to larger.
    step.x = 1;
    step.y = endPoint.y > startPoint.y ? 1 : -1;
    // if (t_ID == checkID) {
    //     printf("Steps: ");
    //     step.print();
    // }

    // Find nearest vertical & horizontal grid lines based on edge direction.
    nextVertical = int(startPoint.x) + 1;
    nextHorizontal = step.y == 1 ? int(startPoint.y) + 1 : int(startPoint.y);
    // if (t_ID == checkID) {
    //     printf("Next horizontal: %d, next vertical: %d\n", nextHorizontal, nextVertical);
    // }

    gradient = (endPoint.y - startPoint.y) / (endPoint.x - startPoint.x);
    edgeLength = sqrt(pow(endPoint.x - startPoint.x, 2) + pow(endPoint.y - startPoint.y, 2));

    // Find intersection with nearest vertical & find tMax.
    intersectY = startPoint.y + (gradient * (nextVertical - startPoint.x));
    distance = sqrt(pow(nextVertical - startPoint.x, 2) + pow(intersectY - startPoint.y, 2));
    // Check if nearest vertical is in range of the edge.
    tMax.x = (distance > edgeLength) ? edgeLength : distance;
    // tMax.x = distance;

    // Find intersection with nearest horizontal & find tMax.
    intersectX = ((nextHorizontal - startPoint.y) / gradient) + startPoint.x;
    distance = sqrt(pow(intersectX - startPoint.x, 2) + pow(nextHorizontal - startPoint.y, 2));
    // Check if nearest horizontal is in range of the edge.
    tMax.y = (distance > edgeLength) ? edgeLength : distance;
    // tMax.y = distance;

    // if (t_ID == checkID) {
    //     printf("Gradient: %f\n", gradient);
    //     printf("Intersection with vertical at: (%d, %f)\n", nextVertical, intersectY);
    //     printf("Intersection with horizontal at: (%f, %d)\n", intersectX, nextHorizontal);
    //     printf("tMax: ");
    //     tMax.print();
    // }

    tDelta.x = edgeLength / (endPoint.x - startPoint.x);
    tDelta.y = edgeLength / fabs(endPoint.y - startPoint.y);

    // Edge traversal, we traverse using the startPoint to save memory.
    while (startPoint.x < endCell.x + 1 && checkYLimit(startPoint.y, endCell.y, step.y))
    {
        poly.setMatrixXY((int)startPoint.x, (int)startPoint.y, PARTIAL_COLOR);
        // if (t_ID == checkID) {
        //     printf("Painted (%d, %d)\n", (int)startPoint.x, (int)startPoint.y);
        //     printf("Current point: ");
        //     startPoint.print();
        //     printf("X check: %d, Y check: %d\n",
        //         (startPoint.x < (endCell.x + 1)),
        //         checkYLimit(startPoint.y, endCell.y, step.y)
        //     );
        // }

        if (tMax.x < tMax.y)
        {
            startPoint.x += step.x;
            tMax.x += tDelta.x;
        }
        else
        {
            startPoint.y += step.y;
            tMax.y += tDelta.y;
        }

        // if (t_ID == checkID) {
        //     printf("Moved point to: ");
        //     startPoint.print();
        // }
    }
}

__device__ bool isPointInsidePolygon(GPUPolygon &poly, GPUPoint testPoint)
{
    GPUPoint edgeStart, edgeEnd;

    bool isInside = false;
    // int intersections = 0;

    // Loop all edges
    for (int j = 0; j < poly.size-1; j++)
    {
        // j = i (start)
        // i = j+1 (end)
        edgeStart = poly.points[j];
        edgeEnd = poly.points[j+1];

        // Check intersection with current edge
        if (((edgeEnd.y > testPoint.y) != (edgeStart.y > testPoint.y)) &&
            (testPoint.x < (edgeStart.x - edgeEnd.x) * (testPoint.y - edgeStart.y) /
            (edgeStart.y - edgeEnd.y) + edgeStart.x))
        {
            isInside = !isInside;
            // intersections++;
        }
    }

    return isInside;
}

__global__ void fillPolygonPerPixel(GPUPolygon &poly, int matrixSize)
{
    GPUPoint edgeStart, edgeEnd;
    GPUPoint testPoint;

    // ID of current thread in reference to all created threads in kernel
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The number of threads in kernel
    int stride = blockDim.x * gridDim.x;

    // Run through the whole rasterization matrix using a grid-stride loop
    for (int pixelID = index; pixelID < matrixSize; pixelID += stride)
    {
        if (poly.matrix[pixelID] == PARTIAL_COLOR) { continue; }

        // Find current pixel coordinates
        testPoint.x = pixelID % poly.mbrWidth;
        testPoint.y = pixelID / poly.mbrWidth;

        poly.matrix[pixelID] =
            (isPointInsidePolygon(poly, testPoint)) ? FULL_COLOR : EMPTY_COLOR;
    }
}

__device__ void floodFillSector(
    GPUPolygon &poly, GPUPoint sectorMin, GPUPoint sectorMax,
    GPUPoint fillPoint, int fillColor
) {
    GPUStack stack;
    GPUPoint currPoint;
    bool hasSpanAbove, hasSpanBelow;

    stack.push(fillPoint.x, fillPoint.y);
    
    while (stack.hasItems())
    {
        currPoint = stack.pop();

        // Go to the start of the line to begin painting.
        while (
            currPoint.x >= sectorMin.x &&
            poly.getMatrixXY(currPoint.x, currPoint.y) == UNCERTAIN_COLOR
        ) {
            currPoint.x--;
        }
        currPoint.x++;
        hasSpanAbove = hasSpanBelow = false;

        // Paint row & check above and below for new points.
        while (
            currPoint.x <= sectorMax.x &&
            poly.getMatrixXY(currPoint.x, currPoint.y) == UNCERTAIN_COLOR
        ) {
            poly.setMatrixXY(currPoint.x, currPoint.y, fillColor);

            // Mark the start of an uncertain span below current point.
            if (
                !hasSpanBelow &&
                currPoint.y > sectorMin.y &&
                poly.getMatrixXY(currPoint.x, currPoint.y - 1) == UNCERTAIN_COLOR
            ) {
                stack.push(currPoint.x, currPoint.y - 1);
                hasSpanBelow = true;
            }
            // Mark the end of an uncertain span below current point.
            else if (
                hasSpanBelow &&
                currPoint.y > sectorMin.y &&
                poly.getMatrixXY(currPoint.x, currPoint.y - 1) != UNCERTAIN_COLOR
            ) {
                hasSpanBelow = false;
            }

            // Mark the start of an uncertain span above current point.
            if (
                !hasSpanAbove &&
                currPoint.y < sectorMax.y &&
                poly.getMatrixXY(currPoint.x, currPoint.y + 1) == UNCERTAIN_COLOR
            ) {
                stack.push(currPoint.x, currPoint.y + 1);
                hasSpanAbove = true;
            }
            // Mark the end of an uncertain span above current point.
            else if (
                hasSpanAbove &&
                currPoint.y < sectorMax.y &&
                poly.getMatrixXY(currPoint.x, currPoint.y + 1) != UNCERTAIN_COLOR
            ) {
                hasSpanAbove = false;
            }

            currPoint.x++;
        }
    }
}

__global__ void floodFillPolygonInSectors(GPUPolygon &poly, int xSectors, int ySectors)
{
    GPUPoint sectorSize;
    GPUPoint sectorMin, sectorMax, currPoint;
    int fillColor;

    sectorSize.x = ceil(poly.mbrWidth / double(xSectors));
    sectorSize.y = ceil(poly.mbrHeight / double(ySectors));

    // ID of current thread in reference to all created threads in kernel.
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;

    // Find the first and last cell of each sector.
    sectorMin.x = (t_ID % xSectors) * sectorSize.x;
    sectorMin.y = (t_ID / xSectors) * sectorSize.y;
    sectorMax.x = sectorMin.x + sectorSize.x - 1;
    sectorMax.y = sectorMin.y + sectorSize.y - 1;

    // Crop edge sectors if needed.
    if (sectorMax.x >= poly.mbrWidth) {
        sectorMax.x = poly.mbrWidth - 1;
    }
    if (sectorMax.y >= poly.mbrHeight) {
        sectorMax.y = poly.mbrHeight - 1;
    }

    // Loop sector points to find not filled ones.
    for (int y = sectorMin.y; y <= sectorMax.y; y++)
    {
        for (int x = sectorMin.x; x <= sectorMax.x; x++)
        {
            if (poly.getMatrixXY(x, y) == UNCERTAIN_COLOR)
            {
                currPoint.x = x;
                currPoint.y = y;

                fillColor = (isPointInsidePolygon(poly, currPoint)) ? FULL_COLOR : EMPTY_COLOR;
                
                // Visualize sectors
                // fillColor = (t_ID % 2) * 2;

                floodFillSector(poly, sectorMin, sectorMax, currPoint, fillColor);
            }
        }
    }
}

void CUDARasterize(GPUPolygon &poly)
{
    // Pointers used by the device.
    GPUPolygon *poly_D;
    GPUPoint *points_D;
    int *matrix_D;

    int matrixSize = poly.mbrHeight * poly.mbrWidth;
    int matrixThreads = (matrixSize > 1024) ? 1024 : matrixSize;

    // Copy polygon to device
    cudaMalloc((void **)&poly_D, sizeof(GPUPolygon));
    cudaMemcpy(poly_D, &poly, sizeof(GPUPolygon), cudaMemcpyHostToDevice);

    // Copy points to device
    cudaMalloc((void **)&points_D, poly.size * sizeof(GPUPoint));
    cudaMemcpy(points_D, poly.points, poly.size * sizeof(GPUPoint), cudaMemcpyHostToDevice);

    // Copy polygon rasterization matrix to device
    size_t mSize = matrixSize * sizeof(int);
    cudaMalloc((void **)&matrix_D, mSize);
    cudaMemcpy(matrix_D, poly.matrix, mSize, cudaMemcpyHostToDevice);

    // Set device polygon's pointers to copied points & matrix
    cudaMemcpy(&(poly_D->points), &points_D, sizeof(GPUPoint *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(poly_D->matrix), &matrix_D, sizeof(int *), cudaMemcpyHostToDevice);

    // Rasterize polygon using per pixel checks
    preparePolygonMatrix<<<1, matrixThreads>>>(*poly_D, matrixSize);
    rasterizeBorder<<<1, poly.size-1>>>(*poly_D);
    fillPolygonPerPixel<<<1, matrixThreads>>>(*poly_D, matrixSize);
    printPolygon<<<1, 1>>>(*poly_D);

    cudaDeviceSynchronize();

    // Rasterize polygon using flood fill per sector
    preparePolygonMatrix<<<1, matrixThreads>>>(*poly_D, matrixSize);
    rasterizeBorder<<<1, poly.size-1>>>(*poly_D);
    floodFillPolygonInSectors<<<1, 25>>>(*poly_D, 5, 5);
    printPolygon<<<1, 1>>>(*poly_D);

    cudaDeviceSynchronize();

    cudaFree(poly_D);
    cudaFree(points_D);
    cudaFree(matrix_D);
}

void calculateMBR(GPUPolygon &poly)
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

void normalizePoints(GPUPolygon &poly)
{
    for (int i = 0; i < poly.size; i++)
    {
        poly.points[i] = GPUPoint(
            poly.hilbertPoints[i].x - poly.hMin.x,
            poly.hilbertPoints[i].y - poly.hMin.y);
    }
}

__host__ void loadPolygonsFromCSV(int startLine, int endLine, std::vector<GPUPolygon> &polygons)
{
    std::ifstream fin;
    std::string line, token, coordToken;
    std::vector<GPUPoint> points;
    int polyID;
    double x, y;

    fin.open(MAPPED_CSV);

    if (!fin.good()) {
        printf("ERROR: dataset file could not be opened.\n");
        return;
    }

    // Skip lines until the desired starting line.
    for (int i = 0; i < startLine; i++)
    {
        getline(fin, line);
        printf("Skipped\n");
    }
    
    // Read & parse lines with polygons.
    for (int i = startLine; i < endLine + 1; i++)
    {
        points.clear();

        getline(fin, line);
        // printf("Line: %s\n", line.c_str());
        std::stringstream lineStream(line);

        // Parse polygon ID at beginning of the line.
        getline(lineStream, token, ',');
        // printf("Token: %s\n", token.c_str());
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
        
        // Copy the points vector in an array
        // GPUPoint pointsArr[points.size()];
        // std::copy(points.begin(), points.end(), pointsArr);

        // printf("Polygon ID: %d\n", polyID);
        // for (int i = 0; i < points.size(); i++)
        // {
        //     points[i].print();
        // }
        
        // Add parsed polygon to list.
        polygons.push_back(GPUPolygon(polyID, points.size(), &points[0]));
    }
    // printf("In function: \t%p\n", &(polygons[0].hilbertPoints[0]));
    // polygons[0].print();
}

int main(void)
{
    std::vector<GPUPolygon> polygons;
    // GPUPolygon* polygonsArr;
    // Create a test polygon
    GPUPoint testPoints[5];
    testPoints[0] = GPUPoint(1.5, 1.5);
    testPoints[1] = GPUPoint(11.5, 21.5);
    testPoints[2] = GPUPoint(31.5, 31.5);
    testPoints[3] = GPUPoint(21.5, 11.5);
    testPoints[4] = GPUPoint(1.5, 1.5);

    GPUPolygon testPoly = GPUPolygon(1, 5, testPoints);
    calculateMBR(testPoly);
    // testPoly.mbrWidth = 5;
    // testPoly.mbrHeight = 3;
    normalizePoints(testPoly);
    testPoly.matrix = new int[testPoly.mbrWidth * testPoly.mbrHeight];

    // testPoly.hilbertPoints[0].print();
    // testPoly.print();
    // testPoints[0] = GPUPoint(100, 100);
    // printf("Hey!\n");
    // testPoly.hilbertPoints[0].print();
    // testPoly.print();
    // testPoints[0].print();
    // testPoly.printMatrix();

    // CUDARasterize(testPoly);
    // TODO move rasterization matrix to Hilbert space

    // std::vector<GPUPoint> points;
    // points.push_back(GPUPoint(2, 3.7));
    // points[0].print();

    loadPolygonsFromCSV(0, 0, polygons);
    // printf("In main: \t%p\n", &(polygons[0].hilbertPoints[0]));
    // for (int i = 0; i < polygons[0].size; i++)
    // {
    //     polygons[0].hilbertPoints[i].print();
    // }
    // polygons[0].hilbertPoints[polygons[0].size-1].print(); 

    calculateMBR(polygons[0]);
    normalizePoints(polygons[0]);
    // polygons[0].matrix = new int[polygons[0].mbrWidth * polygons[0].mbrHeight];
    // polygons[0].print();
    // polygons[0].printMatrix();
    CUDARasterize(polygons[0]);

    return 0;
}