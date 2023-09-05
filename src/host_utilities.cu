/**
 * @file host_utilities.cu
 * @brief This file contails __host__ functions that are used by the CPU.
 */

#include "host_utilities.cuh"

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

    fin.open(MAPPED_CSV);

    if (!fin.good()) {
        printf("ERROR: dataset file could not be opened.\n");
        return;
    }

    // Skip lines until the desired starting line.
    for (int i = 0; i < startLine; i++)
    {
        getline(fin, line);
        // printf("Skipped\n");
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

__host__ GPUPolygon createTestPoly(bool normalize)
{
    GPUPoint testPoints[] = {
        GPUPoint(1.5, 1.5),
        GPUPoint(11.5, 21.5),
        GPUPoint(31.5, 31.5),
        GPUPoint(21.5, 11.5),
        GPUPoint(1.5, 1.5)
    };
    GPUPolygon poly = GPUPolygon(1, 5, testPoints);

    calculateMBR(poly);
    poly.matrix = new int[poly.mbrWidth * poly.mbrHeight];

    if (normalize) { normalizePointsCPU(poly); }
    
    return poly;
}