import csv

from containers import Point, Polygon

SAMPLE_CSV = '../T1NA_fixed.csv'
# Sample min/max
S_MAX = Point(-66.8854, 49.3844)
S_MIN = Point(-124.849, 24.5214)
# Hilbert space min/max
H_MAX = Point(65535, 65535) # 65535 = 2^16 - 1
H_MIN = Point(0, 0)


def readPolygon(id: int, csvPath: str):
    """Read polygon from CSV and parse it."""

    polyFile = open(csvPath)
    polyReader = csv.reader(polyFile)

    # Skip id lines in polygons csv
    for _ in range(id):
        next(polyReader)
    polyID, *polyPoints = next(polyReader)

    polyFile.close()

    # Parse points
    parsedPoints = [Point(eval(x[0]), eval(x[1])) for x in
                    [point.split() for point in polyPoints]]

    # print('ID:', polyID, '\nPoints:', parsedPoints)

    return Polygon(parsedPoints)


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

    hilbertPoly = Polygon(hilbertPoints)

    # Give 1 pixel buffer around MBR if possible
    if hilbertPoly.minX > 0:
        hilbertPoly.minX -= 1
    else:
        hilbertPoly.minX = 0
    
    if hilbertPoly.minY > 0:
        hilbertPoly.minY -= 1
    else:
        hilbertPoly.minY = 0

    if hilbertPoly.maxX < H_MAX.x:
        hilbertPoly.maxX += 1
    else:
        hilbertPoly.maxX = H_MAX.x
        
    if hilbertPoly.maxY < H_MAX.y:
        hilbertPoly.maxY += 1
    else:
        hilbertPoly.maxY = H_MAX.y

    # Calculate MBR size (for later use)
    hilbertPoly.mbrWidth = hilbertPoly.maxX - hilbertPoly.minX + 1
    hilbertPoly.mbrHeight = hilbertPoly.maxY - hilbertPoly.minY + 1

    return hilbertPoly


polygon = readPolygon(0, SAMPLE_CSV)
print(polygon)

hilbertPolygon = mapPolygonToHilber(polygon)
print(hilbertPolygon)