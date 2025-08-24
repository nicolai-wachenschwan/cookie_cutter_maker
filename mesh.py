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

def convert_to_o3d(mesh_in: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Converts a Trimesh object to an Open3D TriangleMesh object."""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_in.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_in.faces)
    return mesh_o3d

def decimate_o3d(mesh_o3d: o3d.geometry.TriangleMesh, target_face_count: int) -> o3d.geometry.TriangleMesh:
    """Decimates an Open3D TriangleMesh object."""
    return mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_face_count)

def convert_from_o3d(mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Converts an Open3D TriangleMesh object back to a Trimesh object."""
    vertices_out = np.asarray(mesh_o3d.vertices)
    faces_out = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices_out, faces=faces_out)

def decimate_mesh(
    mesh_in: trimesh.Trimesh, 
    target_face_ratio: float = 0.5
) -> trimesh.Trimesh:
    """
    Reduces the number of faces of a Trimesh object using Open3D.

    Args:
        mesh_in (trimesh.Trimesh): The original Trimesh object.
        target_face_ratio (float): The desired ratio of faces in the resulting mesh.

    Returns:
        trimesh.Trimesh: The reduced Trimesh object.
    """
    original_face_count = len(mesh_in.faces)
    target_face_count = int(original_face_count * target_face_ratio)

    if original_face_count <= target_face_count:
        print("The face count is already smaller than or equal to the target. Original mesh is returned.")
        return mesh_in

    mesh_o3d = convert_to_o3d(mesh_in)
    mesh_out_o3d = decimate_o3d(mesh_o3d, target_face_count)
    mesh_out = convert_from_o3d(mesh_out_o3d)
    
    return mesh_out
