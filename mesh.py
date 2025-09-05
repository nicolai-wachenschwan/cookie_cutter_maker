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

    # Filtere Faces mit z-Normal > 0 und Centroid z < 0.5
    if not mesh.is_empty:
        face_normals = mesh.face_normals
        face_centroids = mesh.triangles.mean(axis=1)
        
        # Maske für zu löschende Faces
        mask_delete = (face_normals[:, 2] > 1e-6) & (face_centroids[:, 2] < 0.5)
        
        # Behalte nur die Faces, die nicht gelöscht werden sollen
        faces_to_keep = mesh.faces[~mask_delete]
        
        # Erstelle ein neues Mesh mit den gefilterten Faces
        # Dies entfernt auch unreferenzierte Vertices
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces_to_keep)
        mesh.remove_unreferenced_vertices()

    return mesh


def create_base_triangulation(surface_mesh: trimesh.Trimesh, z_threshold: float = 0.5) -> trimesh.Trimesh:
    """
    Erstellt eine triangulierte Grundfläche aus allen Vertices, die unter z_threshold liegen.
    Filtert die Faces der Triangulierung, um nur die zu behalten, die von oben durch das Oberflächennetz verdeckt werden.

    Args:
        surface_mesh (trimesh.Trimesh): Das ursprüngliche Oberflächennetz.
        z_threshold (float): Schwellenwert für die Z-Koordinate.

    Returns:
        trimesh.Trimesh: Neues Mesh mit triangulierter und gefilterter Grundfläche.
    """
    # Finde alle Vertices mit Z < threshold
    base_mask = surface_mesh.vertices[:, 2] < z_threshold
    if not np.any(base_mask):
        print("Warnung: Keine Vertices unter dem Z-Schwellwert gefunden, erstelle leeres Mesh.")
        return trimesh.Trimesh()

    base_vertices = surface_mesh.vertices[base_mask]

    if len(base_vertices) < 3:
        print(f"Warnung: Nur {len(base_vertices)} Vertices gefunden, Triangulierung nicht möglich")
        return trimesh.Trimesh()

    # Projiziere die Vertices auf die XY-Ebene (Z=0)
    points_2d = base_vertices[:, :2]

    try:
        # Delaunay-Triangulierung in 2D
        tri = Delaunay(points_2d)
        all_faces = tri.simplices

        # Erstelle neue Vertices mit Z=0 für eine ebene Grundfläche
        base_vertices_flat = np.column_stack([points_2d, np.zeros(len(points_2d))])

        # Erstelle einen Ray-Intersector für das Oberflächennetz
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(surface_mesh)

        # Filtere Faces basierend auf der Sichtbarkeit von oben
        valid_faces = []
        for face in all_faces:
            # Berechne den Mittelpunkt (Schwerpunkt) des Dreiecks
            v1, v2, v3 = base_vertices_flat[face]
            centroid = (v1 + v2 + v3) / 3.0
            
            # Ray von knapp über dem Schwerpunkt in Z-Richtung
            ray_origin = centroid + np.array([0, 0, 1e-4])
            ray_direction = np.array([0, 0, 1])

            # Prüfe, ob der Ray das Oberflächennetz von unten trifft
            # Wir schauen von unten nach oben, also muss der Strahl etwas treffen
            if intersector.intersects_any([ray_origin], [ray_direction]):
                valid_faces.append(face)

        if not valid_faces:
            print("Warnung: Keine gültigen Faces nach dem Ray-Casting-Filter gefunden.")
            return trimesh.Trimesh()

        faces = np.array(valid_faces)

        # Erstelle das neue Mesh
        base_mesh = trimesh.Trimesh(vertices=base_vertices_flat, faces=faces)
        
        # Bereinige das Mesh
        base_mesh.remove_unreferenced_vertices()
        base_mesh.remove_duplicate_faces()
        
        base_mesh.invert()

        return base_mesh

    except ImportError:
        print("Fehler: scipy ist erforderlich für die Delaunay-Triangulierung")
        return trimesh.Trimesh()
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return trimesh.Trimesh()

def merge_meshes(surface_mesh, base_mesh):
  """
  Merges two meshes into a single mesh.
  """
  return trimesh.util.concatenate(surface_mesh, base_mesh)

def get_transforms(heightmap_array, params):
    """Calculates the scale and center transformations."""
    if heightmap_array is None:
        return None, None

    # Find the bounding box of non-zero elements in the heightmap
    non_zero_coords = np.argwhere(heightmap_array > 0)
    if non_zero_coords.size == 0:
        return None, None  # No content in the heightmap

    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)
    
    # The size of the content in pixels
    content_size_pixels = max_coords - min_coords
    
    # Create a temporary mesh to calculate the center
    temp_mesh = create_mesh_from_heightmap(heightmap_array)
    if temp_mesh.is_empty:
        return None, None

    temp_mesh.process()

    # Determine the scale factor to fit the target size
    target_max_mm = params.get("target_max", 100.0)+2*params.get("w_rim", 5.0)  # in mm
    
    # content_size_pixels is [row, col], which corresponds to [y, x]
    # We want to scale based on the larger of the x or y dimension
    if content_size_pixels[1] == 0 or content_size_pixels[0] == 0:
        scale_factor = 1.0
    else:
        scale_factor = target_max_mm / max(content_size_pixels[1], content_size_pixels[0])

    z_scale_mm = params['h_max'] / 255.0
    scale_transform = trimesh.transformations.compose_matrix(scale=[scale_factor, scale_factor, z_scale_mm])

    center = temp_mesh.bounds.mean(axis=0)
    center_transform = trimesh.transformations.translation_matrix(-center)

    return scale_transform, center_transform


def scale_and_center_mesh(mesh, scale_transform, center_transform):
    """Scales and centers the mesh."""
    if mesh.is_empty:
        return None

    mesh.process()
    mesh.apply_transform(center_transform)
    mesh.apply_transform(scale_transform)

    return mesh

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
    triangulated_base = create_base_triangulation(surface_mesh, z_threshold=0.5)

    # 4. Merge the meshes
    if triangulated_base and not triangulated_base.is_empty:
        merged_mesh = merge_meshes(surface_mesh, triangulated_base)
    else:
        merged_mesh = surface_mesh


    # Clean up the mesh after adding new faces
    trimesh.repair.fill_holes(merged_mesh)
    merged_mesh.remove_duplicate_faces()
    merged_mesh.remove_unreferenced_vertices()
    
    merged_mesh.fix_normals()
    merged_mesh.process(validate=True)
    trimesh.repair.fill_holes(merged_mesh)

    # Get unique edges and their counts
    unique_edges, counts = np.unique(merged_mesh.edges_sorted, axis=0, return_counts=True)
    
    # Non-manifold edges are those that do not appear exactly twice
    non_manifold_mask = counts != 2
    non_manifold_edges = unique_edges[non_manifold_mask]

    print(f"len(non-manifold-detected): {len(non_manifold_edges)}")

    shared_vertex_count = 0
    for i in range(len(non_manifold_edges)):
        for j in range(i + 1, len(non_manifold_edges)):
            edge1 = non_manifold_edges[i]
            edge2 = non_manifold_edges[j]
            if len(set(edge1) & set(edge2)) > 0:
                shared_vertex_count += 1

    print(f"Number of non-manifold edge pairs sharing at least one vertex: {shared_vertex_count}")

    # --- Find loops ---
    loops = []
    if len(non_manifold_edges) > 0:
        from trimesh.graph import connected_components

        # Adjacency list for non-manifold edges
        adj = {v: [] for v in np.unique(non_manifold_edges)}
        for u, v in non_manifold_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Find loops by traversing
        visited_nodes = set()
        for start_node in adj:
            if start_node not in visited_nodes:
                q = [(start_node, [start_node])]
                visited_in_path = {start_node}

                while q:
                    curr_node, path = q.pop(0)

                    for neighbor in adj[curr_node]:
                        if neighbor == path[-2] if len(path) > 1 else False:
                            continue # Don't go back immediately

                        if neighbor == start_node and len(path) > 2:
                            loops.append(path)
                            for node in path:
                                visited_nodes.add(node)
                            q = [] # End search for this component
                            break
                        
                        if neighbor not in visited_in_path:
                            visited_in_path.add(neighbor)
                            new_path = path + [neighbor]
                            q.append((neighbor, new_path))

    # --- Count loops ---
    loop_counts = {}
    if loops:
        for loop in loops:
            length = len(loop)
            loop_counts[length] = loop_counts.get(length, 0) + 1

    print("How many boundary loops where found?")
    if not loop_counts:
        print("No loops found.")
    else:
        for length, count in loop_counts.items():
            if length == 3:
                print(f"- {count} triangles")
            elif length == 4:
                print(f"- {count} quads")
            else:
                print(f"- {count} loops with {length} edges")

    # --- Add faces for triangular loops ---
    new_faces = []
    if loops:
        for loop in loops:
            if len(loop) == 3:
                new_faces.append(loop)

    if new_faces:
        print(f"Adding {len(new_faces)} new faces for the triangular holes.")
        merged_mesh.faces = np.vstack([merged_mesh.faces, new_faces])
        merged_mesh.process() # Re-process the mesh after adding faces
        merged_mesh.fix_normals()

    return merged_mesh


def generate_insert_mesh(insert_map: np.ndarray, outside_mask: np.ndarray, params: dict) -> trimesh.Trimesh:
    """
    Generates a manifold 3D mesh for the insert by extruding the outline.
    """
    # 1. Create the top surface from the heightmap
    insert_surface = create_mesh_from_heightmap(insert_map, cv2.bitwise_not(outside_mask))
    if insert_surface.is_empty:
        return trimesh.Trimesh()

    # 2. Get the ordered boundary vertices of the surface
    outline = insert_surface.outline()
    if not outline:
        print("Warning: Could not determine mesh outline. Returning surface mesh.")
        return insert_surface
    
    try:
        outline_indices = outline.entities[0].points
        surface_boundary_vertices = insert_surface.vertices[outline_indices]
    except (IndexError, AttributeError):
        # Fallback for different outline structures
        try:
            outline_indices = outline[0]
            surface_boundary_vertices = insert_surface.vertices[outline_indices]
        except (IndexError, TypeError):
            print("Warning: Could not extract outline indices. Returning surface mesh.")
            return insert_surface

    if len(surface_boundary_vertices) < 3:
        print("Warning: Not enough boundary vertices to create a solid mesh.")
        return insert_surface

    # 3. Create vertices for the base by translating the boundary down
    ppmm = params.get("ppmm", 3.77)  # 96dpi as fallback
    base_thickness_mm = 3.0
    base_thickness_pixels = base_thickness_mm * ppmm
    z_translation = -base_thickness_pixels
    base_boundary_vertices = surface_boundary_vertices.copy()
    base_boundary_vertices[:, 2] += z_translation

    # 4. Combine all vertices
    num_surface_vertices = len(insert_surface.vertices)
    all_vertices = np.vstack([insert_surface.vertices, base_boundary_vertices])

    # 5. Create side faces (the extrusion)
    side_faces = []
    num_boundary_vertices = len(outline_indices)
    base_indices_offset = num_surface_vertices

    for i in range(num_boundary_vertices):
        p1_idx = outline_indices[i]
        p2_idx = outline_indices[(i + 1) % num_boundary_vertices]
        p3_idx = base_indices_offset + i
        p4_idx = base_indices_offset + ((i + 1) % num_boundary_vertices)

        # Create two triangles for each quad of the side wall
        side_faces.append([p1_idx, p2_idx, p4_idx])
        side_faces.append([p1_idx, p4_idx, p3_idx])

    # 6. Create base faces by triangulating the base polygon
    try:
        # Project to 2D for Delaunay triangulation
        base_polygon_2d = base_boundary_vertices[:, :2]
        tri = Delaunay(base_polygon_2d)
        base_triangles = tri.simplices

        # Offset indices to match the combined vertex list
        base_faces_unfiltered = base_triangles + base_indices_offset
        
        # Filter faces based on the mask
        valid_faces = []
        h, w = outside_mask.shape
        for face in base_faces_unfiltered:
            # Calculate the centroid of the triangle
            v1, v2, v3 = all_vertices[face]
            centroid = (v1 + v2 + v3) / 3.0
            cx, cy = int(centroid[0]), int(centroid[1])

            # Check if the centroid is within the mask
            if 0 <= cy < h and 0 <= cx < w and outside_mask[cy, cx] == 0:
                valid_faces.append(face)
        
        base_faces = np.array(valid_faces)

        # Ensure faces are pointing downwards (outward from the mesh)
        base_faces = base_faces[:, ::-1]
    except Exception as e:
        print(f"Warning: Delaunay triangulation for base failed: {e}")
        base_faces = np.empty((0, 3), dtype=np.int64)

    # 7. Combine all faces
    all_faces = np.vstack([insert_surface.faces, side_faces, base_faces])

    # 8. Create the final mesh
    final_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    
    # 9. Process and return
    final_mesh.fill_holes()
    final_mesh.fix_normals()
    final_mesh.process()

    return final_mesh