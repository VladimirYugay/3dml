"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """
    
    a, b, c = vertices[faces[:, 0], :], vertices[faces[:, 1], :], vertices[faces[:, 2], :]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    prob = areas / areas.sum() 
    
    
    ids = np.random.choice(range(faces.shape[0]), size=n_points, p=prob)
    
    a, b, c = a[ids, :], b[ids, :], c[ids, :]
    r1 = np.random.uniform(0, 1, (ids.shape[0]))
    r2 = np.random.uniform(0, 1, (ids.shape[0]))
    
    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2 
    
    pts = u[:, None] * a + v[:, None] * b + w[:, None] * c
    
    
    
    return pts
