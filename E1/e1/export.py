"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """
    with open(path, 'w') as file:
        for v in vertices:
            print("v {} {} {}".format(str(v[0]), str(v[1]), str(v[2])), file=file)
        for f in faces:
            print("f {} {} {}".format(str(f[0] + 1), str(f[1] + 1), str(f[2] + 1)), file=file)


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """
    with open(path, 'w') as file:
        for v in pointcloud:
            print("v {} {} {}".format(str(v[0]), str(v[1]), str(v[2])), file=file)
    