from pathlib import Path
import json

import numpy as np
import torch


class ShapeNetParts(torch.utils.data.Dataset):
    num_classes = 50  # We have 50 parts classes to segment
    num_points = 1024
    dataset_path = Path("e2/data/shapenetcore_partanno_segmentation_benchmark_v0/")  # path to point cloud data
    class_name_mapping = json.loads(Path("e2/data/shape_parts_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())
    part_id_to_overall_id = json.loads(Path.read_text(Path(__file__).parent.parent / 'data' / 'partid_to_overallid.json'))

    def __init__(self, split):
        assert split in ['train', 'val', 'overfit']

        self.items = Path(f"e2/data/splits/shapenet_parts/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        item = self.items[index]

        pointcloud, segmentation_labels = ShapeNetParts.get_point_cloud_with_labels(item)

        return {
            'points': pointcloud,
            'segmentation_labels': segmentation_labels
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['segmentation_labels'] = batch['segmentation_labels'].to(device)

    @staticmethod
    def get_point_cloud_with_labels(shapenet_id):
        """
        Utility method for reading a ShapeNet point cloud from disk, reads points from pts files on disk as 3d numpy arrays, together with their per-point part labels
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: tuple: a numpy array representing the point cloud, in shape 3x1024, and the segmentation labels, as numpy array in shape 1024
        """
        category_id, shape_id = shapenet_id.split('/')

        points_path = str(ShapeNetParts.dataset_path / category_id / 'points'  / shape_id) + '.pts'
        seg_path = str(ShapeNetParts.dataset_path / category_id / 'points_label' / shape_id) + '.seg'
        points = np.loadtxt(points_path, dtype=np.float32)
        labels = np.loadtxt(seg_path, dtype=np.uint8)
        ids = np.random.choice(points.shape[0], 1024)
        points = points[ids]
        labels = labels[ids]
        global_labels = [None] * 1024 
        for i, label in enumerate(labels):
            global_labels[i] = ShapeNetParts.part_id_to_overall_id[category_id + '_' + str(label)]
        return points.T, np.array(global_labels)
