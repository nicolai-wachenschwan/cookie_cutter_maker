import numpy as np
import trimesh
import io
import pyvista as pv

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

def decimate_mesh(mesh, target_reduction):
    """Reduces the number of faces in the mesh by a target reduction factor."""
    if mesh.is_empty or target_reduction >= 1.0:
        return mesh

    # Wrap the trimesh object in a PyVista PolyData object
    pv_mesh = pv.wrap(mesh)

    # Decimate the mesh
    decimated_pv_mesh = pv_mesh.decimate(target_reduction)

    # Convert back to a trimesh object
    vertices = decimated_pv_mesh.points
    faces = decimated_pv_mesh.faces.reshape((-1, 4))[:, 1:4]
    simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return simplified_mesh
