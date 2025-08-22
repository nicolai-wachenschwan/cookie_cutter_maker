import numpy as np
import trimesh
import io
import pyvista as pv

def generate_3d_model(heightmap, params):
    """Generates a 3D model from the heightmap using Trimesh and returns a PyVista mesh."""
    padded_map = np.pad(heightmap, pad_width=1, mode='constant', constant_values=0)
    matrix = np.zeros((padded_map.shape[0], padded_map.shape[1], 256), dtype=bool)
    for r in range(padded_map.shape[0]):
        for c in range(padded_map.shape[1]):
            height = padded_map[r, c]
            if height > 0:
                matrix[r, c, :height] = True

    voxel_grid = trimesh.voxel.VoxelGrid(matrix)
    mesh = voxel_grid.marching_cubes

    # Handle cases where the mesh is empty
    if mesh.is_empty:
        return None, None

    mesh.process()

    ppmm = max(heightmap.shape) / params['target_max'] if params['target_max'] > 0 else 1
    if ppmm == 0:
        return None, None  # Avoid division by zero
    pixel_width_mm = 1.0 / ppmm
    z_scale_mm = params['h_max'] / 255.0
    scale_transform = trimesh.transformations.compose_matrix(scale=[pixel_width_mm, pixel_width_mm, z_scale_mm])
    center_transform = trimesh.transformations.translation_matrix(-mesh.bounds.mean(axis=0))
    mesh.apply_transform(center_transform)
    mesh.apply_transform(scale_transform)

    with io.BytesIO() as f:
        mesh.export(f, file_type='stl')
        f.seek(0)
        stl_data = f.read()

    vertices = mesh.vertices
    faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)).flatten()
    pv_mesh = pv.PolyData(vertices, faces)

    return stl_data, pv_mesh
