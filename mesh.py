import numpy as np
import trimesh
import cv2
# import open3d as o3d
from scipy.spatial import Delaunay

def create_mesh_from_heightmap(heightmap: np.ndarray, mask: np.ndarray = None) -> trimesh.Trimesh:
    """
    Erzeugt ein 3D-Netz direkt aus einer 2D-Heightmap.
    Verarbeitet nur Pixel, die in der binären Maske als 255 markiert sind.

    Args:
        heightmap (np.ndarray): Ein 2D-Array, bei dem jeder Wert die Höhe darstellt.
        mask (np.ndarray, optional): Binäre Maske, nur Pixel mit Wert 255 werden verarbeitet.
                                   Falls None, werden alle Pixel verarbeitet.
    Returns:
        trimesh.Trimesh: Das resultierende 3D-Netz.
    """
    # Dimensionen der Heightmap auslesen
    h, w = heightmap.shape

    # Maske für gültige Pixel
    if mask is not None:
        # Stelle sicher, dass die Maske die richtige Größe hat
        assert mask.shape == heightmap.shape, "Maske muss die gleiche Größe wie die Heightmap haben"
        valid_mask = mask == 255
    else:
        # Falls keine Maske übergeben wird, verwende alle Pixel
        valid_mask = np.ones((h, w), dtype=bool)

    # Erzeuge ein Gitter von X- und Y-Koordinaten
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    # Nur gültige Vertices extrahieren
    valid_x = xx[valid_mask]
    valid_y = yy[valid_mask]
    valid_z = heightmap[valid_mask]

    # Erstelle die Vertices nur für gültige Punkte
    vertices = np.stack([valid_x, valid_y, valid_z], axis=1)

    # Erstelle eine Mapping-Matrix für die neuen Vertex-Indizes
    # -1 bedeutet ungültiger Vertex
    vertex_map = np.full((h, w), -1, dtype=int)
    vertex_map[valid_mask] = np.arange(np.sum(valid_mask))

    # --- Erzeugung der Faces (Dreiecke) ---
    faces = []

    # Iteriere über alle möglichen Quads (außer am Rand)
    for i in range(h - 1):
        for j in range(w - 1):
            # Indizes der vier Ecken des aktuellen Quads
            idx_tl = vertex_map[i, j]       # Top-left
            idx_tr = vertex_map[i, j + 1]   # Top-right
            idx_bl = vertex_map[i + 1, j]   # Bottom-left
            idx_br = vertex_map[i + 1, j + 1] # Bottom-right

            # Überprüfe, ob alle vier Ecken gültige Vertices haben
            if idx_tl >= 0 and idx_tr >= 0 and idx_bl >= 0 and idx_br >= 0:
                # Erstes Dreieck: top-left, top-right, bottom-left
                faces.append([idx_tl, idx_tr, idx_bl])
                # Zweites Dreieck: top-right, bottom-right, bottom-left
                faces.append([idx_tr, idx_br, idx_bl])

    # Konvertiere zu numpy array
    faces = np.array(faces)

    # Erstelle das Trimesh-Objekt (nur wenn Faces vorhanden sind)
    if len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        # Fallback: leeres Mesh wenn keine gültigen Faces
        mesh = trimesh.Trimesh()

    return mesh


def create_base_triangulation(mesh: trimesh.Trimesh, mask: np.ndarray = None, z_threshold: float = 1.0) -> trimesh.Trimesh:
    """
    Erstellt eine triangulierte Grundfläche aus allen Vertices, die auf Z=0 oder Z<threshold liegen.
    Filtert die Faces basierend auf deren Mittelpunkt - nur Faces deren Schwerpunkt in maskierten Bereichen liegt.

    Args:
        mesh (trimesh.Trimesh): Das ursprüngliche Mesh
        mask (np.ndarray, optional): Binäre Maske zur Filterung der Faces
        z_threshold (float): Schwellwert für Z-Koordinate (Standard: 1.0)

    Returns:
        trimesh.Trimesh: Neues Mesh mit triangulierter und gefilterter Grundfläche
    """
    # Finde alle Vertices mit Z < threshold
    base_mask = mesh.vertices[:, 2] < z_threshold
    base_vertices = mesh.vertices[base_mask]

    # Falls weniger als 3 Vertices, kann nicht trianguliert werden
    if len(base_vertices) < 3:
        print(f"Warnung: Nur {len(base_vertices)} Vertices gefunden, Triangulierung nicht möglich")
        return trimesh.Trimesh()

    # Projiziere die Vertices auf die XY-Ebene (Z=0)
    points_2d = base_vertices[:, :2]  # Nur X und Y Koordinaten

    try:
        # Delaunay-Triangulierung in 2D
        tri = Delaunay(points_2d)
        all_faces = tri.simplices

        # Erstelle neue Vertices mit Z=0 für eine ebene Grundfläche
        base_vertices_flat = np.column_stack([points_2d, np.zeros(len(points_2d))])

        # Filtere Faces basierend auf deren Mittelpunkt (falls Maske vorhanden)
        if mask is not None:
            valid_faces = []
            h, w = mask.shape

            for face in all_faces:
                # Berechne den Mittelpunkt (Schwerpunkt) des Dreiecks
                v1, v2, v3 = base_vertices_flat[face]
                centroid = (v1 + v2 + v3) / 3.0
                cx, cy = int(centroid[0]), int(centroid[1])

                # Prüfe ob Mittelpunkt innerhalb der Maskengrenzen und in maskiertem Bereich liegt
                if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
                    valid_faces.append(face)

            if len(valid_faces) == 0:
                print("Warnung: Keine gültigen Faces nach Masken-Filterung gefunden")
                return trimesh.Trimesh()

            faces = np.array(valid_faces)
        else:
            faces = all_faces

        # Erstelle das neue Mesh
        base_mesh = trimesh.Trimesh(vertices=base_vertices_flat, faces=faces)
        base_mesh.invert()

        return base_mesh

    except ImportError:
        print("Fehler: scipy ist erforderlich für die Delaunay-Triangulierung")
        return trimesh.Trimesh()

def merge_meshes(surface_mesh, base_mesh):
  """
  Merges two meshes into a single mesh.
  """
  return trimesh.util.concatenate(surface_mesh, base_mesh)

def scale_and_center_mesh(mesh, params, center=None):
    """Scales and centers the mesh."""
    if mesh.is_empty:
        return None

    mesh.process()

    ppmm = params.get("ppmm", 3.77)  # 96dpi as fallback
    if ppmm == 0:
        return None  # Avoid division by zero
    pixel_width_mm = 1.0 / ppmm
    z_scale_mm = params['h_max'] / 255.0
    scale_transform = trimesh.transformations.compose_matrix(scale=[pixel_width_mm, pixel_width_mm, z_scale_mm])

    if center is None:
        center = mesh.bounds.mean(axis=0)

    center_transform = trimesh.transformations.translation_matrix(-center)
    mesh.apply_transform(center_transform)
    mesh.apply_transform(scale_transform)

    return mesh

# def convert_to_o3d(mesh_in: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
#     """Converts a Trimesh object to an Open3D TriangleMesh object."""
#     mesh_o3d = o3d.geometry.TriangleMesh()
#     mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_in.vertices)
#     mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_in.faces)
#     return mesh_o3d

# def decimate_o3d(mesh_o3d: o3d.geometry.TriangleMesh, target_face_count: int) -> o3d.geometry.TriangleMesh:
#     """Decimates an Open3D TriangleMesh object."""
#     return mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_face_count)

# def convert_from_o3d(mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
#     """Converts an Open3D TriangleMesh object back to a Trimesh object."""
#     vertices_out = np.asarray(mesh_o3d.vertices)
#     faces_out = np.asarray(mesh_o3d.triangles)
#     return trimesh.Trimesh(vertices=vertices_out, faces=faces_out)

# def decimate_mesh(
#     mesh_in: trimesh.Trimesh,
#     target_face_ratio: float = 0.5
# ) -> trimesh.Trimesh:
#     """
#     Reduces the number of faces of a Trimesh object using Open3D.

#     Args:
#         mesh_in (trimesh.Trimesh): The original Trimesh object.
#         target_face_ratio (float): The desired ratio of faces in the resulting mesh.

#     Returns:
#         trimesh.Trimesh: The reduced Trimesh object.
#     """
#     original_face_count = len(mesh_in.faces)
#     target_face_count = int(original_face_count * target_face_ratio)

#     if original_face_count <= target_face_count:
#         print("The face count is already smaller than or equal to the target. Original mesh is returned.")
#         return mesh_in

#     mesh_o3d = convert_to_o3d(mesh_in)
#     mesh_out_o3d = decimate_o3d(mesh_o3d, target_face_count)
#     mesh_out = convert_from_o3d(mesh_out_o3d)
    
#     return mesh_out


def generate_mesh(heightmap: np.ndarray, params: dict) -> trimesh.Trimesh:
    """
    Generates a 3D mesh from a heightmap using the new direct method.
    """
    # 1. Create a mask from the heightmap
    _, threshed_map = cv2.threshold(heightmap, 1, 255, cv2.THRESH_BINARY)
    dilated_map = cv2.dilate(threshed_map, np.ones((3, 3), np.uint8), iterations=1)

    # 2. Convert the heightmap to a mesh
    surface_mesh = create_mesh_from_heightmap(heightmap, dilated_map)

    # 3. Create a triangulated base
    # Using a small z_threshold to select vertices close to the base plane.
    triangulated_base = create_base_triangulation(surface_mesh, mask=dilated_map, z_threshold=1.0)

    # 4. Merge the meshes
    if triangulated_base and not triangulated_base.is_empty:
        merged_mesh = merge_meshes(surface_mesh, triangulated_base)
    else:
        merged_mesh = surface_mesh

    # 5. Fix normals and process
    merged_mesh.fix_normals()
    merged_mesh.process()

    return merged_mesh


def generate_insert_mesh(insert_map: np.ndarray, outside_mask: np.ndarray, params: dict) -> trimesh.Trimesh:
    """
    Generates a 3D mesh for the insert.
    """
    if np.sum(insert_map) == 0:
        return trimesh.Trimesh()

    # Create a heightmap for the insert
    h_max = params.get("h_max", 15.0)
    insert_height = h_max + 1
    insert_heightmap = (insert_map > 0) * insert_height

    # Generate the top surface
    insert_surface = create_mesh_from_heightmap(insert_heightmap)

    # Create the base
    # The mask for the base should be outside_mask bitwise_and with inverted insert map
    inverted_insert_map = cv2.bitwise_not(insert_map)
    base_mask = cv2.bitwise_and(outside_mask, inverted_insert_map)

    # dilate base mask to connect parts
    kernel = np.ones((3,3), np.uint8)
    base_mask_dilated = cv2.dilate(base_mask, kernel, iterations=2)

    # Generate a flat base plane using create_base_triangulation
    # We pass the surface mesh to get the vertices, and the base_mask to filter
    insert_base = create_base_triangulation(insert_surface, mask=base_mask_dilated, z_threshold=insert_height + 1)

    # Extrude the base by 3mm
    # We need a polygon for that. We can get it from the contours of the base mask
    contours, _ = cv2.findContours(base_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return trimesh.Trimesh()

    # Assuming the largest contour is the one we want to extrude
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = trimesh.path.polygons.Polygon(largest_contour.reshape(-1, 2))

    # Extrude the polygon downwards by 3mm
    # The `generate_base` function in the prompt is a bit ambiguous.
    # The description sounds more like an extrusion of the whole insert shape.
    # Let's try to extrude the insert shape itself.

    contours_insert, _ = cv2.findContours(insert_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_insert:
        return trimesh.Trimesh()

    # Combine all contours into one polygon if there are multiple
    all_points = np.vstack([c for c in contours_insert])

    # Find the convex hull of all points to get the outer boundary
    hull = cv2.convexHull(all_points)

    polygon_insert = trimesh.path.polygons.Polygon(hull.reshape(-1, 2))

    # Extrude the insert polygon
    # The height of the extrusion is 3mm, as requested.
    # We need to consider the scaling factor (ppmm)
    ppmm = params.get("ppmm", 3.77)
    extrusion_height = 3.0 # The prompt says 3mm

    # Let's create the mesh by extruding the polygon
    # The base of the insert should be at the same level as the cutter's rim
    h_rim = params.get("h_rim", 2.0)

    # The insert should be extruded downwards.
    # We can achieve this by creating a path and then extruding.
    # The path will be a line from z=h_rim to z=h_rim-3

    # Actually, let's rethink the extrusion.
    # The request says "extrude the base 3mm (direction: increase overall thickness)"
    # This is a bit confusing. I will interpret it as the insert having a solid base of 3mm thickness.

    # So, I'll generate the top surface at h_max+1.
    # And a bottom surface at h_max+1 - 3mm.
    # And then stitch them together.

    # Let's use the extrusion method, it's cleaner.
    # I will extrude the insert polygon by 3mm.

    # The height of the insert surface is `h_max + 1`. Let's stick to that.
    # The extrusion should probably be from the base of the cookie cutter up to a certain height.
    # Let's reconsider the prompt: "extrude the base 3mm (direction: increase overall thickness)"
    # This might mean that the base of the insert, which connects to the cutter, should be 3mm thick.

    # Let's try a different approach. I will create the insert as a separate object.
    # Top surface at h_max+1. Bottom surface at 0. Then connect them.

    # The prompt is "extrude the base 3mm".
    # This probably means the part of the insert that connects to the cutter body.

    # Let's try to implement it like this:
    # 1. Generate the insert shape as a 2D polygon.
    # 2. Extrude it to a height of `h_max + 1`.
    # 3. Create a base for it, which is another extrusion of a different shape, with 3mm height.

    # Let's go with the initial interpretation.
    # 1. Top surface from heightmap.
    # 2. Base from triangulation.
    # 3. And then what to do with the 3mm extrusion?

    # "extrude the base 3mm (direction: increase overall thickness)"
    # This could mean that the `insert_base` mesh should be extruded.
    # `trimesh` doesn't have a direct mesh extrusion function.

    # Let's go back to polygon extrusion.
    # I'll extrude the `insert_map` contour.

    # The prompt says "use the same generate-base function". This is `create_base_triangulation`.
    # "as mask we use the „outside_mask“ of the original heightmap, bitwise_and with inverted insert map."
    # This base is for connecting the insert to the main body.

    # Let's try to build the insert part by part.
    # Part 1: The stamp/insert itself.
    # This is an extrusion of the `insert_map`'s contour.
    # The prompt says "scale the height to h_max+1". This is the height of the stamp.

    extrusion_height_insert = params.get("h_max") + 1.0

    # We need to find the contours of the insert_map
    contours_insert, _ = cv2.findContours(insert_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    insert_parts = []
    for contour in contours_insert:
        if cv2.contourArea(contour) > 2: # Ignore very small contours
             polygon = trimesh.path.polygons.Polygon(contour.reshape(-1, 2))
             # Extrude from z=0 to extrusion_height_insert
             insert_part = trimesh.creation.extrude_polygon(polygon, height=extrusion_height_insert)
             insert_parts.append(insert_part)

    if not insert_parts:
        return trimesh.Trimesh()

    insert_mesh = trimesh.util.concatenate(insert_parts)

    # Part 2: The connecting base.
    # "extrude the base 3mm (direction: increase overall thickness)"
    # The base is defined by the mask: `outside_mask` AND NOT `insert_map`.

    inverted_insert_map = cv2.bitwise_not(insert_map)
    base_mask = cv2.bitwise_and(outside_mask, inverted_insert_map)

    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours_base, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    base_parts = []
    for contour in contours_base:
        if cv2.contourArea(contour) > 2:
            polygon = trimesh.path.polygons.Polygon(contour.reshape(-1, 2))
            # Extrude by 3mm. The base should be at the bottom.
            # So we extrude from z=0 to z=3.
            base_part = trimesh.creation.extrude_polygon(polygon, height=3.0)
            base_parts.append(base_part)

    if base_parts:
        base_mesh = trimesh.util.concatenate(base_parts)
        # Combine the insert and the base
        full_insert_mesh = trimesh.util.concatenate(insert_mesh, base_mesh)
    else:
        full_insert_mesh = insert_mesh

    full_insert_mesh.fix_normals()
    full_insert_mesh.process()

    return full_insert_mesh
