/**
 * @file kernels.cu
 * @brief This file contails __global__ functions that are used to initiate
 * a new kernel at the GPU.
 */

#include "gpu_containers.cuh"
#include "kernels.cuh"
#include "gpu_utilities.cuh"

__global__ void printPolygon(GPUPolygon &poly)
{
    poly.print();
    poly.printMatrix();
}

__global__ void normalizePoints(GPUPolygon &poly)
{
    // ID of current thread in reference to all created threads in kernel
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;
    // The number of threads in kernel
    int stride = blockDim.x * gridDim.x;

    for (int i = t_ID; i < poly.size; i += stride)
    {
        poly.points[i] = GPUPoint(
            poly.hilbertPoints[i].x - poly.hMin.x,
            poly.hilbertPoints[i].y - poly.hMin.y);
    }
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
        poly.matrix[i] = UNCERTAIN_COLOR;
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // The number of threads in kernel
    int stride = blockDim.x * gridDim.x;

    // Rasterize polygon border using a grid-stride loop.
    for (int t_ID = index; t_ID < poly.size - 1; t_ID += stride)
    {
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
            // printf("Thread %d stopped early at: (%d, %d)\n",
                // t_ID, (int)startCell.x, (int)startCell.y);
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

__global__ void floodFillPolygonInSectors(GPUPolygon &poly, int xSectors, int ySectors)
{
    GPUPoint sectorSize;
    GPUPoint sectorMin, sectorMax, currPoint;
    int fillColor;

    sectorSize.x = ceil(poly.mbrWidth / double(xSectors));
    sectorSize.y = ceil(poly.mbrHeight / double(ySectors));

    // ID of current thread in reference to all created threads in kernel.
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;

    // if (t_ID == 0) {
    //     printf("Sectors on X: %d, Sector size on X: %d\n", xSectors, (int)sectorSize.x);
    //     printf("Sectors on Y: %d, Sector size on Y: %d\n\n", ySectors, (int)sectorSize.y);
    // }

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