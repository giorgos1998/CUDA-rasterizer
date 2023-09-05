/**
 * @file gpu_utilities.cu
 * @brief This file contains __device__ functions that are used only by the GPU
 * kernels.
 */

#include "gpu_utilities.cuh"

__device__ bool checkYLimit(double testPointY, int endCellY, int stepY)
{
    if (stepY > 0) {
        return (testPointY < endCellY + stepY);
    } else {
        return (testPointY > endCellY);
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