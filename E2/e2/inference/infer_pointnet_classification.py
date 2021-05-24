"""Utility for inference using trained networks"""

import torch

from e2.data.shapenet import ShapeNetPoints
from e2.model.pointnet import PointNetClassification


class InferenceHandlerPointNetClassification:
    """Utility for inference using trained PointNet network"""

    def __init__(self, ckpt):
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = PointNetClassification(ShapeNetPoints.num_classes)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()

    def infer_single(self, points):
        """
        Infer class of the shape given its point cloud representation
        :param points: points of shape 3 x 1024
        :return: class category name for the point cloud, as predicted by the model
        """
        input_tensor = torch.from_numpy(points).float().unsqueeze(0)

        # TODO: Predict class
        prediction = self.model(input_tensor)
        class_id = torch.argmax(prediction)
        class_name = ShapeNetPoints.class_name_mapping[ShapeNetPoints.classes[class_id]]

        return class_name
