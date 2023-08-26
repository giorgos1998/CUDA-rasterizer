import csv

from containers import Point, Polygon

SAMPLE_CSV = '../T1NA_fixed.csv'
MAPPED_CSV = '../T1NA_mapped.csv'
# Sample min/max
S_MAX = Point(-66.8854, 49.3844)
S_MIN = Point(-124.849, 24.5214)
# Hilbert space min/max
H_MAX = Point(65535, 65535) # 65535 = 2^16 - 1
H_MIN = Point(0, 0)


def readPolygons(startID: int, endID: int, csvPath: str):
    """Read polygons from CSV and parse it."""

    parsedPolygons = []

    polyFile = open(csvPath)
    polyReader = csv.reader(polyFile)

    # Skip id lines in polygons csv
    for _ in range(startID):
        next(polyReader)
    
    for i in range(startID, endID):
        try:
            polyID, *polyPoints = next(polyReader)
        except StopIteration:
            print(f"Stopped at {i+1} of {endID}")
            exit(1)

        # Parse points
        parsedPoints = [Point(eval(x[0]), eval(x[1])) for x in
                        [point.split() for point in polyPoints]]

        parsedPolygons.append(Polygon(int(polyID), parsedPoints))
        print(f"Read polygon {i+1} of {endID}", end='\r')
    
    polyFile.close()
    print("\nRead all polygons!\n")

    return parsedPolygons


def mapPolygonToHilber(polygon: Polygon):
    """Maps given polygon to Hilbert space."""

    hilbertPoints = []

    for point in polygon.vertices:
        hilbertX = (H_MAX.x / (S_MAX.x - S_MIN.x)) * (point.x - S_MIN.x)
        hilbertY = (H_MAX.y / (S_MAX.y - S_MIN.y)) * (point.y - S_MIN.y)

        if hilbertX < 0: hilbertX = 0
        if hilbertY < 0: hilbertY = 0
        
        if hilbertX >= H_MAX.x + 1: hilbertX = H_MAX.x
        if hilbertY >= H_MAX.y + 1: hilbertY = H_MAX.y

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
    
    polyFile = open(filepath, 'w')
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

# Main 123045 polygons
loadedPolygons = readPolygons(0, 123045, SAMPLE_CSV)
mappedPolygons = []

for i, polygon in enumerate(loadedPolygons):
    mappedPolygons.append(mapPolygonToHilber(polygon))
    print(f"Parsed polygon {i+1} of {len(loadedPolygons)}", end='\r')

print("\nWriting to mapped CSV...")
writePolygonsToCSV(mappedPolygons, MAPPED_CSV)