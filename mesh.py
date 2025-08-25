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

def scale_and_center_mesh(mesh, params):
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
    center_transform = trimesh.transformations.translation_matrix(-mesh.bounds.mean(axis=0))
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
