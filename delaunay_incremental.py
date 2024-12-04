import matplotlib.pyplot as plt
import numpy as np


def dist(a, b):
    """Eucledian distance between two points.

    Args:
        a (tuple): first point
        b (tuple): second point

    Returns:
        float: distance between a and b
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def signed_area(a, b, c):
    """Calculate the signed area of the triangle formed by three points.

    Args:
        a (tuple): first point
        b (tuple): second point
        c (tuple): third point

    Returns:
        float: signed area of the triangle
    """
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2


def in_circle(a, b, c, d):
    """
    Check if a point is inside the circle defined by three points.

    Args:
        a (tuple): first point
        b (tuple): second point
        c (tuple): third point
        d (tuple): point to check

    Raises:
        ValueError: if a, b, and c are collinear

    Returns:
        bool: if d is inside the circle defined by a, b, and c
    """
    t = np.array([[1, 1, 1], [a[0], b[0], c[0]], [a[1], b[1], c[1]]])
    dt = np.linalg.det(t)
    if dt == 0:
        raise ValueError("Points are collinear")
    m = np.array(
        [
            [1, 1, 1, 1],
            [a[0], b[0], c[0], d[0]],
            [a[1], b[1], c[1], d[1]],
            [
                a[0] ** 2 + a[1] ** 2,
                b[0] ** 2 + b[1] ** 2,
                c[0] ** 2 + c[1] ** 2,
                d[0] ** 2 + d[1] ** 2,
            ],
        ]
    )
    dm = np.linalg.det(m)
    return dt * dm <= 1e-6


def fix_orientation(T):
    """
    Since the triangles in the list are kept as sets for easier comparison, when converted back to lists
    they may have a non-standard vertex order. This function fixes that to ensure delauney_in_triangle works correctly.

    Args:
        T (list): list of the vertices of a triangle

    Returns:
        list: the vertices of the triangle in the correct order
    """
    if signed_area(T[0], T[1], T[2]) < 0:
        return (T[0], T[2], T[1])
    return T


def delaunay_in_triangle(T, p):
    """
    Check if a point is inside, outside, or on the edge of a triangle. The triangle must be oriented counter-clockwise.
    We make the distinction between inside and on the edge because we need to add another triangle if the point is on the edge.

    Args:
        T (list): the vertices of the triangle
        p (tuple): the point to check

    Returns:
        bool,bool: if the point is inside the triangle, if the point is on the edge of the triangle
    """
    T = fix_orientation(T)
    for i in range(3):
        area = signed_area(T[i - 1], T[i], p)
        if in_segment(p, T[i - 1], T[i]):
            return (True, True)
        elif area < 0:
            return (False, False)
    return (True, False)


# Carlos' function to test Delaunay criterion
def test_delaunay(triangulacion, puntos):
    """Test que verifica el criterio de Delaunay en cada triángulo de la triangulación."""
    for tri in triangulacion:
        p1, p2, p3 = tri[0], tri[1], tri[2]

        # Comprobamos que no haya otro punto dentro del circuncírculo de este triángulo
        for pto in puntos:
            if pto not in [p1, p2, p3]:
                if in_circle(p1, p2, p3, pto):
                    print(
                        f"Error: El punto {pto} está dentro del circuncírculo del triángulo formado por los puntos {p1}, {p2}, {p3}."
                    )


def in_segment(p, a, b):
    """
    Check if a point is inside the segment defined by two points.

    Args:
        p (tuple): the point to check
        a (tuple): first point of the segment
        b (tuple): second point of the segment

    Returns:
        bool: if the point is on the segment
    """
    return abs(float(dist(p, a)) + float(dist(p, b)) - float(dist(a, b))) < 1e-6


def legalise_side(points, triangles, side, new_point):
    """
    Legalisation procedure for the sides of a triangle. If a point is inside the circumcircle of
    a triangle, we make a flip with the side of the adjacent triangle.

    Args:
        points (list): list of all points up to the current point
        triangles (list): triangles in the triangulation up to the current point
        side (list): side of the triangle to legalise
        new_point (tuple): new point added to the triangulation, used to check if it is inside the circumcircle
    """
    for point in points:
        if (
            in_circle(new_point, side[0], side[1], point)
            and point
            not in [
                side[0],
                side[1],
                new_point,
            ]
            and set((side[0], side[1], point)) in triangles
        ):
            # Adding and removing these triangles is equivalent to flipping the side
            triangles.append(set((side[0], point, new_point)))
            triangles.append(set((side[1], point, new_point)))
            triangles.remove(set((side[0], side[1], point)))
            triangles.remove(set((side[0], side[1], new_point)))
            # After making the flip we need to legalise other sides
            legalise_side(points, triangles, (side[0], point), new_point)
            legalise_side(points, triangles, (side[1], point), new_point)


def delaunay_incremental(S):
    """
    Incremental Delaunay triangulation algorithm. We start with a triangle that contains all points and add points one by one.

    Args:
        S (list): list of points to triangulate

    Returns:
        list: list of sets of tuples, each set represents a triangle, each tuple is a vertex of the triangle
    """
    # Convert all points to tuples to be able to add them to sets
    S = [tuple(p) for p in S]
    # triangles is a list of sets of tuples, each set represents a triangle
    triangles = []
    # Create the initial triangle that contains all points
    initial_triangle = create_initial_triangle(S)
    triangles.append(set(initial_triangle))
    triangulations = []
    triangulations.append(
        ([tuple(i) for i in triangles], S[:3] + list(initial_triangle))
    )
    for i, new_point in enumerate(S):
        sides_to_legalise = []
        triangles_to_remove = []
        triangles_to_add = []
        # This is used for cases when the point is on one of the sides for it to keep looking
        seen_triangle = False
        for triangle in triangles:
            # We need to convert the set to a list to be able to access the elements
            triangle = list(triangle)
            in_triangle, on_edge = delaunay_in_triangle(triangle, new_point)
            if in_triangle:
                # We will need to remove the old triangle and add three new ones (equivalent to adding three edges)
                triangles_to_remove.append(set(triangle))
                for j in range(3):
                    # If the point is on an edge, we don't add the triangle that contains that edge (it's a segment)
                    if not in_segment(new_point, triangle[j - 1], triangle[j]):
                        triangles_to_add.append(
                            set((triangle[j - 1], triangle[j], new_point))
                        )
                        # We also need to legalise the the edges of the old triangle
                        sides_to_legalise.append((triangle[j - 1], triangle[j]))
                if not on_edge or seen_triangle:
                    break
                else:
                    seen_triangle = True
        # We remove the old triangles and add the new ones
        for triangle in triangles_to_remove:
            triangles.remove(triangle)
        triangles.extend(triangles_to_add)
        # We legalise the sides of the triangles
        for side in sides_to_legalise:
            legalise_side(S[:i] + list(initial_triangle), triangles, side, new_point)
        # We test the Delaunay criterion
        # NOTE: This gives a false positive for when there are points that are just on the circumcircle of another 3 points
        test_delaunay([list(i) for i in triangles], S[: i + 1])
        triangulations.append(
            ([tuple(i) for i in triangles], S[: i + 1] + list(initial_triangle))
        )
        # OPTIONAL: Uncomment to visualise the triangulation at each step
        # visualizar_triangulacion(
        #     [list(i) for i in triangles], S[: i + 1] + list(initial_triangle)
        # )
    # We remove the initial triangle
    # We need to reverse the list of triangles to not skip any when removing
    for triangle in reversed(triangles):
        if triangle.intersection(set(initial_triangle)):
            triangles.remove(triangle)
    triangulations.append(([tuple(i) for i in triangles], S))
    return triangulations


def create_initial_triangle(S, margin=1):
    """
    Create a triangle that contains all points in S. We take the minimum x and y coordinates and the maximum coordinate in
    the direction of the (1,1) vector. We then find the vertices of the triangle determined by these two points. We create it by
    making a vertical and horizontal line from (x_min, y_min) and finding the intersection with the line that goes through (max_rx, max_ry)
    and has (1,1) as a normal vector.

    Args:
        S (list): the list of points
        margin (int, optional): additional margin to make the triangle slightly larger. Defaults to 1.

    Returns:
        tuple: the vertices of the initial triangle
    """
    min_x = min(S, key=lambda p: p[0])[0]
    min_y = min(S, key=lambda p: p[1])[1]
    max_rx = max(S, key=lambda p: p[0] + p[1])[0]
    max_ry = max(S, key=lambda p: p[0] + p[1])[1]

    # Create a triangle that contains all points in S
    p1 = (min_x - margin, min_y - margin)
    p2 = (max_rx + max_ry - min_y + 2 * margin, min_y - margin)
    p3 = (min_x - margin, max_ry + max_rx - min_x + 2 * margin)

    return (p1, p2, p3)


# Function used by carlos to make the visualisation
def calcular_circuncentro(tri):
    # Tomamos los tres vértices del triángulo
    A, B, C = tri

    # Calcular las longitudes de los lados
    AB = np.linalg.norm(B - A)
    BC = np.linalg.norm(C - B)
    CA = np.linalg.norm(A - C)

    # Usamos la fórmula de las coordenadas del circuncentro (intersección de mediatrices)
    # Se asume que A, B y C son arrays numpy de forma (2,)
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))

    ux = (
        (A[0] ** 2 + A[1] ** 2) * (B[1] - C[1])
        + (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1])
        + (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])
    ) / D
    uy = (
        (A[0] ** 2 + A[1] ** 2) * (C[0] - B[0])
        + (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0])
        + (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])
    ) / D

    return np.array([ux, uy])


# Function used by carlos to make the visualisation
def visualizar_triangulacion(triangulacion, puntos):
    puntos = np.array(puntos)
    point_to_index = {tuple(p): i for i, p in enumerate(puntos)}
    # Convert triangles from coordinates to indices
    triangles = []
    for triangle in triangulacion:
        indices = [point_to_index[tuple(vertex)] for vertex in triangle]
        triangles.append(indices)
    triangulacion = np.array(triangles)
    plt.triplot(puntos[:, 0], puntos[:, 1], triangulacion, color="gray")

    # Dibujar los puntos
    plt.plot(puntos[:, 0], puntos[:, 1], "o", color="red")

    # Calcular y dibujar el circuncentro y el círculo circunscrito para cada triángulo
    for t in triangulacion:
        # Vértices del triángulo
        A = puntos[t[0]]
        B = puntos[t[1]]
        C = puntos[t[2]]

        # Calcular el circuncentro
        circuncentro = calcular_circuncentro(np.array([A, B, C]))

        # Calcular el radio del círculo circunscrito (distancia desde el circuncentro a uno de los vértices)
        radio = np.linalg.norm(circuncentro - A)

        # Dibujar el circuncentro
        plt.plot(circuncentro[0], circuncentro[1], "go")  # Circuncentro en verde

        # Dibujar el círculo circunscrito
        circunferencia = plt.Circle(
            circuncentro, radio, color="blue", fill=False, linestyle="dotted"
        )
        plt.gca().add_artist(circunferencia)

    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    S = [[0, -1], [3, 0], [-1, 2], [2, 1], [1, 0], [2, 2]]
    T = delaunay_incremental(S)
    print(T[-1][0])
    # Convert all points to tuples to be able to visualise them (the function expects tuples)
    # T = [tuple(p) for p in T]
    print(T[-1][1])
    visualizar_triangulacion(T[0][0], T[0][1])
