import math
import csv

OUTPUT_FILE = 'regular_polygons.csv'
MBR_MAX = 4096
# CENTER = MBR_MAX / 2
# RADIUS = CENTER - 1.5
MAX_VERTICES = 100

# vertices = int(input("Enter desired number of vertices: "))
id = 1
csvRows = []

print(f"Creating regular polygons \
(Max MBR: {MBR_MAX}, Max vertices: {MAX_VERTICES})")

for vertices in range(3, MAX_VERTICES+1):
    mbr = 4

    while mbr <= MBR_MAX:
        points = []
        center = mbr / 2
        radius = center - 1.5

        # Create regular polygon points.
        for i in range(vertices):
            x = center + radius * math.cos(2 * math.pi * i / vertices)
            y = center + radius * math.sin(2 * math.pi * i / vertices)
            points.append([x, y])

        # Calculate MBR.
        minPointX = 0
        minPointY = 0
        maxPointX = 0
        maxPointY = 0
        minX = points[0][0]
        minY = points[0][1]
        maxX = points[0][0]
        maxY = points[0][1]

        for i, point in enumerate(points):
            if point[0] < minX:
                minX = point[0]
                minPointX = i
            if point[1] < minY:
                minY = point[1]
                minPointY = i
            if point[0] > maxX:
                maxX = point[0]
                maxPointX = i
            if point[1] > maxY:
                maxY = point[1]
                maxPointY = i

        # Add 1 cell buffer to MBR all around.
        minX = int(minX) - 1
        minY = int(minY) - 1
        maxX = int(maxX) + 2
        maxY = int(maxY) + 2

        # Move min and max points to match required MBR.
        if minX > 0:
            points[minPointX][0] -= minX
        if minY > 0:
            points[minPointY][1] -= minY
        if maxX < mbr:
            points[maxPointX][0] += (mbr - maxX)
        if maxY < mbr:
            points[maxPointY][1] += (mbr - maxY)

        # Add 1st point to the end of the list.
        points.append(points[0])

        row = [id]
        row.extend([f"{p[0]} {p[1]}" for p in points])
        csvRows.append(row)

        mbr = mbr * 2
        id += 1

        # print("Dataset points:")
        # print(",".join([f"{p[0]} {p[1]}" for p in points]))

        # print("\nPlot points:")
        # print(",".join([f"({p[0]},{p[1]})" for p in points]))

# Write polygons to CSV file.
print("Writing to CSV...")
polyFile = open(OUTPUT_FILE, 'w')
polyWriter = csv.writer(polyFile)
polyWriter.writerows(csvRows)
polyFile.close()
