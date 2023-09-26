import csv

POLYGONS_CSV = "T1NA_mapped.csv"
# POLYGONS_CSV = "regular_polygons.csv"

target = input("Enter ID of target polygon: ")
points = []

polyFile = open(POLYGONS_CSV)
polyReader = csv.reader(polyFile)

for row in polyReader:
    if (row[0] == target):
        points = [(float(p[0]), float(p[1])) for p in
                  [point.split() for point in row[1:]]]
        break
else:
    print("Could not find polygon.")
    polyFile.close()
    exit(1)

polyFile.close()

minX = points[0][0]
minY = points[0][1]
maxX = points[0][0]
maxY = points[0][1]

for point in points:
    if point[0] < minX:
        minX = point[0]
    if point[1] < minY:
        minY = point[1]
    if point[0] > maxX:
        maxX = point[0]
    if point[1] > maxY:
        maxY = point[1]

minX = int(minX) - 1
minY = int(minY) - 1
maxX = int(maxX) + 2
maxY = int(maxY) + 2

mbr = [f"({minX - minX},{minY - minY})",
       f"({minX - minX},{maxY - minY})",
       f"({maxX - minX},{maxY - minY})",
       f"({maxX - minX},{minY - minY})"]

print(f"Polygon {target}:")
print(f"Size: {len(points)}")
print(f"MBR: {maxX-minX}x{maxY-minY}")

# print("\nPoints:")
# print(",".join([f"({p[0] - minX},{p[1] - minY})" for p in points]))

# print("\nMBR points:")
# print(",".join(mbr))
