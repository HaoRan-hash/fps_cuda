import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import logging
from data_aug import *
import fps_cuda
import time


class S3dis(Dataset):
    def __init__(self, root, split, loop, npoints=24000, voxel_size=0.04, test_area=5, transforms=None):
        super(S3dis, self).__init__()
        self.root = root
        self.split = split
        self.loop = loop
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.transforms = transforms
        self.idx_to_class = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 
                5: 'window', 6: 'door', 7: 'table', 8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'}
        
        room_list = os.listdir(root)
        if split == 'train':
            self.room_list = list(filter(lambda x : f'Area_{test_area}' not in x, room_list))
        else:
            self.room_list = list(filter(lambda x : f'Area_{test_area}' in x, room_list))
    
    def __len__(self):
        return len(self.room_list) * self.loop

    def voxel_grid_sampling(self, pos):
        """
        pos.shape = (n, 3)
        """
        voxel_indices = np.floor(pos / self.voxel_size).astype(np.int64)
        voxel_max = voxel_indices.max(axis=0)
        
        temp = np.ones_like(voxel_max)
        temp[1] = voxel_max[0]
        temp[2] = voxel_max[0] * voxel_max[1]
        
        voxel_hash = (voxel_indices * temp).sum(axis=-1)
        sort_idx = voxel_hash.argsort()
        
        _, counts = np.unique(voxel_hash, return_counts=True)
        if self.split == 'test':   # test时需要的东西和train，val时不同
            return sort_idx, counts
        
        idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + np.random.randint(0, counts.max(), counts.size) % counts
        return sort_idx[idx_select]
    
    def __getitem__(self, index):
        room = os.path.join(self.root, self.room_list[index % len(self.room_list)])
        points = np.load(room)
        
        # 大家都这样做
        points[:, 0:3] = points[:, 0:3] - np.min(points[:, 0:3], axis=0)
        
        if self.split == 'test':
            sort_idx, counts = self.voxel_grid_sampling(points[:, 0:3])
            pos, x, y = points[:, 0:3], points[:, 3:-1], points[:, -1]
            pos, x, y = pos.astype(np.float32), x.astype(np.float32), y.astype(np.int64)
            return pos, x, y, sort_idx, counts
        
        # train, val的流程
        sample_indices = self.voxel_grid_sampling(points[:, 0:3])
        # 再随机采固定个点
        if self.split == 'train':
            sample_indices = np.random.choice(sample_indices, (self.npoints, ))
        pos, x, y = points[sample_indices, 0:3], points[sample_indices, 3:-1], points[sample_indices, -1]
        if self.transforms:
            pos, x = self.transforms(pos, x)
        
        pos, x, y = pos.astype(np.float32), x.astype(np.float32), y.astype(np.int64)
        return pos, x, y


@torch.no_grad()
def fps(points, nsamples):
    """
    points.shape = (b, n, 3)
    return indices.shape = (b, nsamples)
    """
    b, n, _ = points.shape
    device = points.device
    dis = torch.ones((b, n), device=device) * 1e10
    indices = torch.zeros((b, nsamples), device=device, dtype=torch.long)

    for i in range(1, nsamples):
        cur_index = indices[:, i - 1].view(b, 1, 1).expand(-1, -1, 3)
        cur_point = points.gather(1, cur_index)

        temp = (points - cur_point).square().sum(axis=2)
        mask = (temp < dis)
        dis[mask] = temp[mask]

        index = dis.argmax(dim=1)
        dis[list(range(b)), index] = 0
        indices[:, i] = index
    return indices


@torch.no_grad()
def fps2(points, nsamples):
    """
    points.shape = (b, n, 3)
    return indices.shape = (b, nsamples)
    """
    b, n, _ = points.shape
    device = points.device
    dis = torch.ones((b, n), device=device) * 1e10
    indices = torch.zeros((b, nsamples), device=device, dtype=torch.long)
    
    fps_cuda.fps(points, dis, indices)
    return indices


if __name__ == '__main__':
    log_dir = 'test_01.log'
    logging.basicConfig(filename=log_dir, format='%(message)s', level=logging.INFO)

    train_aug = Compose([ColorContrast(p=0.2),
                            PointCloudScaling(0.9, 1.1),
                            PointCloudFloorCentering(),
                            ColorDrop(p=0.2),
                            ColorNormalize()])
    train_dataset = S3dis('/home/lindi/chenhr/threed/data/processed_s3dis', split='train', loop=30, transforms=train_aug)
    train_dataloader = DataLoader(train_dataset, 8, shuffle=False)
    
    device = 'cuda:5'
    pos, _, _ = next(iter(train_dataloader))
    pos = pos.to(device)
#     pos = pos[:, 0:pos.shape[1] // 64, :]
    print(pos.shape)
    
    st = time.time()
    res1 = fps(pos, pos.shape[1] // 4)
    et = time.time()
    print(f'fps1 time: {et-st}')
    logging.info(f'{res1[:, 0:10]}')
    
    st = time.time()
    res2 = fps2(pos, pos.shape[1] // 4)
    et = time.time()
    print(f'fps2 time: {et-st}')
    logging.info(f'{res2[:, 0:10]}')
    
    # # 验证一致性
    # mask = (res1 - res2)
    # logging.info(f'{mask}')
    
