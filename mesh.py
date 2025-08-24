import numpy as np
import trimesh
import io
import pyvista as pv
import open3d as o3d
import numpy as np

def create_voxel_matrix(heightmap, params):
    """Creates a voxel matrix from the heightmap."""
    padded_map = np.pad(heightmap, pad_width=1, mode='constant', constant_values=0)
    matrix = np.zeros((padded_map.shape[0], padded_map.shape[1], 256), dtype=bool)
    for r in range(padded_map.shape[0]):
        for c in range(padded_map.shape[1]):
            height = padded_map[r, c]
            if height > 0:
                matrix[r, c, :height] = True
    return matrix

def create_mesh_from_voxel_matrix(matrix):
    """Generates a mesh from the voxel matrix using marching cubes."""
    voxel_grid = trimesh.voxel.VoxelGrid(matrix)
    mesh = voxel_grid.marching_cubes
    return mesh

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

def decimate_mesh(
    mesh_in: trimesh.Trimesh, 
    target_face_count: int
) -> trimesh.Trimesh:
    """
    Reduziert die Anzahl der Faces eines Trimesh-Objekts mit Open3D.

    Args:
        mesh_in (trimesh.Trimesh): Das ursprüngliche Trimesh-Objekt.
        target_face_count (int): Die gewünschte Anzahl an Faces im Ergebnis-Mesh.

    Returns:
        trimesh.Trimesh: Das reduzierte Trimesh-Objekt.
    """
    # 1. Überprüfen, ob eine Reduzierung überhaupt notwendig ist
    if len(mesh_in.faces) <= target_face_count:
        print("Die Face-Anzahl ist bereits kleiner oder gleich dem Ziel. Original-Mesh wird zurückgegeben.")
        return mesh_in

    # 2. Konvertierung von Trimesh zu Open3D TriangleMesh
    # Open3D benötigt Vertices und Faces in speziellen Vektor-Formaten.
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_in.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_in.faces)

    # 3. Dezimierung mit Open3D durchführen
    # Die Funktion 'simplify_quadric_decimation' ist schnell und qualitätserhaltend.
    mesh_out_o3d = mesh_o3d.simplify_quadric_decimation(
        target_number_of_triangles=target_face_count
    )

    # 4. Rückkonvertierung von Open3D zu Trimesh
    # Die Geometriedaten werden aus dem Open3D-Objekt extrahiert 
    # und in NumPy-Arrays umgewandelt, die Trimesh versteht.
    vertices_out = np.asarray(mesh_out_o3d.vertices)
    faces_out = np.asarray(mesh_out_o3d.triangles)
    
    return trimesh.Trimesh(vertices=vertices_out, faces=faces_out)
