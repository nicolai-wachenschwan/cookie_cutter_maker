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

def scale_and_center_mesh(mesh, params, scale_transform=None, center_transform=None):
    """Scales and centers the mesh."""
    if mesh.is_empty:
        return None

    mesh.process()

    ppmm = params.get("ppmm", 3.77)  # 96dpi as fallback
    if ppmm == 0:
        return None  # Avoid division by zero
    pixel_width_mm = 1.0 / ppmm
    z_scale_mm = params['h_max'] / 255.0
    if scale_transform is None:
        scale_transform = trimesh.transformations.compose_matrix(scale=[pixel_width_mm, pixel_width_mm, z_scale_mm])

    center = mesh.bounds.mean(axis=0)
    if center_transform is None:
        center_transform = trimesh.transformations.translation_matrix(-center)
    mesh.apply_transform(center_transform)
    mesh.apply_transform(scale_transform)

    return mesh, scale_transform, center_transform

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
    Generates a manifold 3D mesh for the insert with corrected Z-scaling.
    """
    from scipy.spatial import cKDTree

    # Generate the top surface
    insert_surface = create_mesh_from_heightmap(insert_map, cv2.bitwise_not(outside_mask))
    if insert_surface.is_empty:
        return insert_surface

    # --- Z-scaling fix ---
    desired_h_max = 16.0
    current_h_max = params.get('h_max', 16.0)
    if current_h_max > 0:
        z_correction_factor = desired_h_max / current_h_max
        insert_surface.vertices[:, 2] *= z_correction_factor
    
    # --- Manifold mesh generation ---
    try:
        # Find the ordered boundary vertices of the surface
        outline = insert_surface.outline()
        # Map outline vertices to their indices in the main mesh's vertex list
        tree = cKDTree(insert_surface.vertices)
        _, surface_boundary_indices = tree.query(outline.vertices)
    except Exception as e:
        print(f"Warning: Could not determine mesh outline, falling back to simple merge. Error: {e}")
        z_scale_mm = desired_h_max / 255.0
        insert_base = create_base_triangulation(insert_surface, mask=cv2.bitwise_not(outside_mask), z_threshold=0.5)
        if not insert_base.is_empty:
            insert_base.apply_translation((0, 0, -3 * z_scale_mm))
            merged_mesh = merge_meshes(insert_surface, insert_base)
        else:
            merged_mesh = insert_surface
        merged_mesh.fix_normals()
        merged_mesh.process()
        return merged_mesh

    if len(surface_boundary_indices) < 3:
        print("Warning: Not enough boundary vertices to create a solid mesh.")
        return insert_surface

    # Create vertices for the base
    surface_boundary_vertices = insert_surface.vertices[surface_boundary_indices]
    base_boundary_vertices = surface_boundary_vertices.copy()

    base_thickness_mm = params.get('base_thickness', 3.0)
    if current_h_max > 0:
        base_z = -base_thickness_mm * (255.0 / current_h_max)
    else:
        base_z = 0  # Avoid division by zero
    base_boundary_vertices[:, 2] = base_z

    # Combine vertices
    num_surface_vertices = len(insert_surface.vertices)
    all_vertices = np.vstack([insert_surface.vertices, base_boundary_vertices])

    # Create side faces
    side_faces = []
    num_boundary_vertices = len(surface_boundary_indices)
    base_indices_offset = num_surface_vertices

    for i in range(num_boundary_vertices):
        p1_idx = surface_boundary_indices[i]
        p2_idx = surface_boundary_indices[(i + 1) % num_boundary_vertices]
        p3_idx = base_indices_offset + i
        p4_idx = base_indices_offset + ((i + 1) % num_boundary_vertices)

        side_faces.append([p1_idx, p4_idx, p2_idx])
        side_faces.append([p1_idx, p3_idx, p4_idx])

    # Create base faces by triangulating the base polygon
    try:
        # Project to 2D for Delaunay triangulation
        base_polygon_2d = base_boundary_vertices[:, :2]
        tri = Delaunay(base_polygon_2d)
        base_triangles = tri.simplices

        # Offset indices to match the combined vertex list
        base_faces = base_triangles + base_indices_offset
        # Ensure faces are pointing downwards (outward from the mesh)
        base_faces = base_faces[:, ::-1]
    except Exception as e:
        print(f"Warning: Delaunay triangulation for base failed: {e}")
        base_faces = np.empty((0, 3), dtype=np.int64)

    # Combine all faces
    all_faces = np.vstack([insert_surface.faces, side_faces, base_faces])

    # Create the final mesh
    final_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    final_mesh.fix_normals()
    final_mesh.process()

    return final_mesh

