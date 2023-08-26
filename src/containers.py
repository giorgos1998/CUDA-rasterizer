class Point:

    def __init__(self, x: float, y: float) -> None:
        """Creates a point with x, y coordinates."""
        self.x = x
        self.y = y


class Polygon:

    def __init__(self, id: int, vertices: list[Point]) -> None:
        """Creates a polygon using a list of points."""

        if len(vertices) < 3:
            raise ValueError("Cannot create polygon with less than 3 vertices")

        self.id = id
        self.vertices = vertices

        # self.minX = self.vertices[0].x
        # self.maxX = self.vertices[0].x
        # self.minY = self.vertices[0].y
        # self.maxY = self.vertices[0].y

        # for vertex in self.vertices:
        #     if vertex.x < self.minX:
        #         self.minX = vertex.x
        #     if vertex.y < self.minY:
        #         self.minY = vertex.y
        #     if vertex.x > self.maxX:
        #         self.maxX = vertex.x
        #     if vertex.y > self.maxY:
        #         self.maxY = vertex.y

        # self.mbrWidth = self.maxX - self.minX + 1
        # self.mbrHeight = self.maxY - self.minY + 1
        # self.mbr = (Point(minX, minY), Point(maxX, maxY))
        

    def __str__(self) -> str:
        string = f"Polygon ID: {self.id}"
        for point in self.vertices:
            string += f"\n({point.x}, {point.y})"
        return string