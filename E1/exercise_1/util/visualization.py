""" Visualization utilities """
from pathlib import Path

import numpy as np
import k3d
from matplotlib import cm, colors
import trimesh


def visualize_occupancy(occupancy_grid, flip_axes=False):
    point_list = np.concatenate([c[:, np.newaxis] for c in np.where(occupancy_grid)], axis=1)
    visualize_pointcloud(point_list, 1, flip_axes, name='occupancy_grid')


def visualize_pointcloud(point_cloud, point_size, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(np.float32), point_size=point_size, color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


def visualize_mesh(vertices, faces, flip_axes=False):
    plot = k3d.plot(name='points', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        vertices[:, 2] = vertices[:, 2] * -1
        vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
    plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()


def visualize_sdf(sdf: np.array, filename: Path) -> None:
    assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2], "SDF grid has to be of cubic shape"
    print(f"Creating SDF visualization for {sdf.shape[0]}^3 grid ...")

    voxels = np.stack(np.meshgrid(range(sdf.shape[0]), range(sdf.shape[1]), range(sdf.shape[2]))).reshape(3, -1).T

    sdf[sdf < 0] /= np.abs(sdf[sdf < 0]).max() if np.sum(sdf < 0) > 0 else 1.
    sdf[sdf > 0] /= sdf[sdf > 0].max() if np.sum(sdf < 0) > 0 else 1.
    sdf /= -2.

    norm = colors.Normalize(vmin=-1, vmax=1)

    colormap = cm.get_cmap('seismic')
    num_vertices = voxels.shape[0]

    cube_vertices = np.zeros(shape=[8 * num_vertices, 3])
    cube_vertex_colors = np.zeros(shape=[8 * num_vertices, 3])
    for voxel_idx, voxel in enumerate(voxels):
        scale_factor = sdf[tuple(voxel)]
        cube_vertices[voxel_idx * 8 + 0] = voxel + np.array([-.25, -.25, -.25]) * scale_factor
        cube_vertices[voxel_idx * 8 + 1] = voxel + np.array([.25, -.25, -.25]) * scale_factor
        cube_vertices[voxel_idx * 8 + 2] = voxel + np.array([-.25, .25, -.25]) * scale_factor
        cube_vertices[voxel_idx * 8 + 3] = voxel + np.array([.25, .25, -.25]) * scale_factor
        cube_vertices[voxel_idx * 8 + 4] = voxel + np.array([-.25, -.25, .25]) * scale_factor
        cube_vertices[voxel_idx * 8 + 5] = voxel + np.array([.25, -.25, .25]) * scale_factor
        cube_vertices[voxel_idx * 8 + 6] = voxel + np.array([-.25, .25, .25]) * scale_factor
        cube_vertices[voxel_idx * 8 + 7] = voxel + np.array([.25, .25, .25]) * scale_factor
        cube_vertex_colors[voxel_idx * 8 + 0] = colormap(norm(scale_factor))[:3]
        cube_vertex_colors[voxel_idx * 8 + 1] = colormap(norm(scale_factor))[:3]
        cube_vertex_colors[voxel_idx * 8 + 2] = colormap(norm(scale_factor))[:3]
        cube_vertex_colors[voxel_idx * 8 + 3] = colormap(norm(scale_factor))[:3]
        cube_vertex_colors[voxel_idx * 8 + 4] = colormap(norm(scale_factor))[:3]
        cube_vertex_colors[voxel_idx * 8 + 5] = colormap(norm(scale_factor))[:3]
        cube_vertex_colors[voxel_idx * 8 + 6] = colormap(norm(scale_factor))[:3]
        cube_vertex_colors[voxel_idx * 8 + 7] = colormap(norm(scale_factor))[:3]

    cube_faces = np.zeros(shape=[12 * num_vertices, 3])
    for faces_idx in range(num_vertices):
        cube_faces[faces_idx * 12 + 0] = np.array([1, 0, 2]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 1] = np.array([2, 3, 1]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 2] = np.array([5, 1, 3]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 3] = np.array([3, 7, 5]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 4] = np.array([4, 5, 7]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 5] = np.array([7, 6, 4]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 6] = np.array([0, 4, 6]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 7] = np.array([6, 2, 0]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 8] = np.array([3, 2, 6]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 9] = np.array([6, 7, 3]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 10] = np.array([5, 4, 0]) + faces_idx * 8
        cube_faces[faces_idx * 12 + 11] = np.array([0, 1, 5]) + faces_idx * 8

    mesh = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces,
                           vertex_colors=cube_vertex_colors, process=False)
    mesh.export(str(filename))
    print(f"Exported to {filename}")


def visualize_shape_alignment(R=None, t=None):
    mesh_input = trimesh.load(Path(__file__).parent.parent / "resources" / "mesh_input.obj")
    mesh_target = trimesh.load(Path(__file__).parent.parent / "resources" / "mesh_target.obj")
    plot = k3d.plot(name='aligment', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    input_vertices = np.array(mesh_input.vertices)
    if not (R is None or t is None):
        t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, mesh_input.vertices.shape[0]))
        input_vertices = (R @ input_vertices.T + t_broadcast).T
    plt_mesh_0 = k3d.mesh(input_vertices.astype(np.float32), np.array(mesh_input.faces).astype(np.uint32), color=0xd00d0d)
    plt_mesh_1 = k3d.mesh(np.array(mesh_target.vertices).astype(np.float32), np.array(mesh_target.faces).astype(np.uint32), color=0x0dd00d)
    plot += plt_mesh_0
    plot += plt_mesh_1
    plt_mesh_0.shader = '3d'
    plt_mesh_1.shader = '3d'
    plot.display()
