from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        input_sdf = np.clip(input_sdf, -self.truncation_distance, self.truncation_distance)
        target_df = np.clip(target_df, -self.truncation_distance, self.truncation_distance)
        
        signs = np.where(input_sdf >= 0, 1, -1)
        input_sdf = np.abs(input_sdf)
        input_sdf = np.concatenate(
            (input_sdf[None, ...], signs[None, ...]), 
            axis=0)
        
        target_df = np.log(target_df + 1)
        
        input_sdf = input_sdf.astype(np.single)
        target_df = target_df.astype(np.single)        

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):

        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_df'] = batch['target_df'].to(device)


    @staticmethod
    def get_shape_sdf(shapenet_id):
                
        dx, dy, dz = np.fromfile(
            str(ShapeNet.dataset_sdf_path / shapenet_id) + '.sdf', 
            dtype=np.uint64, 
            count=3)
        
        data = np.fromfile(
            str(ShapeNet.dataset_sdf_path / shapenet_id) + '.sdf', 
            dtype=np.single, 
            count=dx * dy * dz,
            offset=8)

        return data.reshape(dx, dy, dz)


    @staticmethod
    def get_shape_df(shapenet_id):
        
        dx, dy, dz = np.fromfile(
            str(ShapeNet.dataset_df_path / shapenet_id) + '.df', 
            dtype=np.uint64, 
            count=3)
        
        data = np.fromfile(
            str(ShapeNet.dataset_df_path / shapenet_id) + '.df', 
            dtype=np.single, 
            count=dx * dy * dz,
            offset=8)
        
        return data.reshape(dx, dy, dz)
