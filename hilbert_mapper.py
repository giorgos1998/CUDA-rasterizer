import csv

from containers import Point, Polygon

SAMPLE_CSV = '../T1NA_fixed.csv'
H_MAX = Point(2**16, 2**16)
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

    orderN = 1
    hilbertPoints = []

    for point in polygon.vertices:
        hilbertX = ((orderN-1) / (H_MAX.x - H_MIN.x)) * (point.x - H_MIN.x)
        hilbertY = ((orderN-1) / (H_MAX.y - H_MIN.y)) * (point.y - H_MIN.y)

        if hilbertX < 0: hilbertX = 0
        if hilbertY < 0: hilbertY = 0
        
        if hilbertX >= orderN: hilbertX = orderN-1
        if hilbertY >= orderN: hilbertY = orderN-1

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

    if hilbertPoly.maxX < orderN-1:
        hilbertPoly.maxX += 1
    else:
        hilbertPoly.maxX = orderN-1
        
    if hilbertPoly.maxY < orderN-1:
        hilbertPoly.maxY += 1
    else:
        hilbertPoly.maxY = orderN-1

    # Calculate MBR size (for later use)
    hilbertPoly.mbrWidth = hilbertPoly.maxX - hilbertPoly.minX + 1
    hilbertPoly.mbrHeight = hilbertPoly.maxY - hilbertPoly.minY + 1

    return hilbertPoly


polygon = readPolygon(10, SAMPLE_CSV)
print(polygon)

hilbertPolygon = mapPolygonToHilber(polygon)
print(hilbertPolygon)