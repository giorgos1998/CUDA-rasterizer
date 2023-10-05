import csv
import math

from containers import Point, Polygon

SAMPLE_CSV = '../T1NA_fixed.csv'
MAPPED_CSV = '../T1NA_mapped.csv'
POLYGONS = 123045       # T1NA
# POLYGONS = 2252316    # T2NA
# POLYGONS = 3043       # T3NA
BATCH_SIZE = 400000
# Sample min/max
S_MAX = Point(-66.8854, 49.3844)
S_MIN = Point(-124.849, 24.5214)
# Hilbert space min/max
H_MAX = Point(65535, 65535)  # 65535 = 2^16 - 1
H_MIN = Point(0, 0)


def readPolygons(startLine: int, endLine: int, csvPath: str):
    """Read polygons from CSV and parse it (including endLine)."""

    parsedPolygons = []

    polyFile = open(csvPath)
    polyReader = csv.reader(polyFile)

    # Skip id lines in polygons csv
    for _ in range(1, startLine):
        next(polyReader)

    for i in range(startLine, endLine + 1):
        try:
            polyID, *polyPoints = next(polyReader)
        except StopIteration:
            print(f"Stopped after reading {i-1} of {endLine}\n")
            return parsedPolygons

        # Parse points
        parsedPoints = [Point(eval(x[0]), eval(x[1])) for x in
                        [point.split() for point in polyPoints]]

        parsedPolygons.append(Polygon(int(polyID), parsedPoints))
        print(f"Read polygon {i} of {endLine}", end='\r')

    polyFile.close()
    print("\nRead all polygons!\n")

    return parsedPolygons


def mapPolygonToHilber(polygon: Polygon):
    """Maps given polygon to Hilbert space."""

    hilbertPoints = []

    for point in polygon.vertices:
        hilbertX = (H_MAX.x / (S_MAX.x - S_MIN.x)) * (point.x - S_MIN.x)
        hilbertY = (H_MAX.y / (S_MAX.y - S_MIN.y)) * (point.y - S_MIN.y)

        if hilbertX < 0:
            hilbertX = 0
        if hilbertY < 0:
            hilbertY = 0

        if hilbertX >= H_MAX.x + 1:
            hilbertX = H_MAX.x
        if hilbertY >= H_MAX.y + 1:
            hilbertY = H_MAX.y

        hilbertPoints.append(Point(hilbertX, hilbertY))

    hilbertPoly = Polygon(polygon.id, hilbertPoints)

    # Give 1 pixel buffer around MBR if possible
    # if hilbertPoly.minX > 0:
    #     hilbertPoly.minX -= 1
    # else:
    #     hilbertPoly.minX = 0

    # if hilbertPoly.minY > 0:
    #     hilbertPoly.minY -= 1
    # else:
    #     hilbertPoly.minY = 0

    # if hilbertPoly.maxX < H_MAX.x:
    #     hilbertPoly.maxX += 1
    # else:
    #     hilbertPoly.maxX = H_MAX.x

    # if hilbertPoly.maxY < H_MAX.y:
    #     hilbertPoly.maxY += 1
    # else:
    #     hilbertPoly.maxY = H_MAX.y

    # # Calculate MBR size (for later use)
    # hilbertPoly.mbrWidth = hilbertPoly.maxX - hilbertPoly.minX + 1
    # hilbertPoly.mbrHeight = hilbertPoly.maxY - hilbertPoly.minY + 1

    return hilbertPoly


def writePolygonsToCSV(polygons: list[Polygon], filepath: str):

    print("\nWriting to mapped CSV...")

    polyFile = open(filepath, 'a')
    polyWriter = csv.writer(polyFile)

    csvRows = []

    for polygon in polygons:
        newRow = []

        newRow.append(polygon.id)
        for point in polygon.vertices:
            newRow.append(f"{point.x} {point.y}")

        csvRows.append(newRow)

    polyWriter.writerows(csvRows)
    polyFile.close()
    print("Wrote batch to file!\n")


# Main
batches = math.ceil(POLYGONS / BATCH_SIZE)
# Clear file before writing
open(MAPPED_CSV, 'w').close()
print(f"Input file: '{SAMPLE_CSV}', Output file: '{MAPPED_CSV}'\n")

for i in range(batches):
    print(f"=========== Batch {i+1} of {batches} ===========")
    start = (i * BATCH_SIZE) + 1
    end = (i+1) * BATCH_SIZE if (i < batches-1) else POLYGONS

    print(f"Start: {start}, End: {end}")
    loadedPolygons = readPolygons(start, end, SAMPLE_CSV)
    mappedPolygons = []

    for j, polygon in enumerate(loadedPolygons):
        mappedPolygons.append(mapPolygonToHilber(polygon))
        print(f"Mapped polygon {j+1} of {len(loadedPolygons)}", end='\r')

    writePolygonsToCSV(mappedPolygons, MAPPED_CSV)

    loadedPolygons.clear()
    mappedPolygons.clear()

    if len(loadedPolygons) < BATCH_SIZE:
        print("Finished mapping!")
        break
