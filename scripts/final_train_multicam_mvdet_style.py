import argparse
import os
import json
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.models import resnet18, vgg11
from kornia.geometry.transform import warp_perspective
from scipy.stats import multivariate_normal
from scipy.sparse import coo_matrix
from PIL import Image
import cv2
import xml.etree.ElementTree as ET


def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
    image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
    world_coord = project_mat @ image_coord
    world_coord = world_coord[:2, :] / world_coord[2, :]
    return world_coord


def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.delete(project_mat, 2, 1)
    world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    image_coord = project_mat @ world_coord
    image_coord = image_coord[:2, :] / image_coord[2, :]
    return image_coord


class GaussianMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel):
        target = self._target_transform(x, target, kernel)
        return F.mse_loss(x, target)

    def _target_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device),
                              padding=int((kernel.shape[-1] - 1) / 2))
        return target


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


intrinsic_camera_matrix_filenames_wildtrack = [
    'intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
    'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml'
]
extrinsic_camera_matrix_filenames_wildtrack = [
    'extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
    'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml'
]


class Wildtrack(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'Wildtrack'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]
        self.num_cam, self.num_frame = 7, 2000
        self.indexing = 'ij'
        self.worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)]
        )

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 480

    def get_worldgrid_from_worldcoord(self, world_coord):
        coord_x, coord_y = world_coord
        grid_x = (coord_x + 300) / 2.5
        grid_y = (coord_y + 900) / 2.5
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(
            os.path.join(intrinsic_camera_path, intrinsic_camera_matrix_filenames_wildtrack[camera_i]),
            flags=cv2.FILE_STORAGE_READ
        )
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(
            os.path.join(self.root, 'calibrations', 'extrinsic',
                         extrinsic_camera_matrix_filenames_wildtrack[camera_i])
        ).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array([float(x) for x in rvec], dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array([float(x) for x in tvec], dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix


intrinsic_camera_matrix_filenames_multiviewx = [
    'intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml',
    'intr_Camera4.xml', 'intr_Camera5.xml', 'intr_Camera6.xml'
]
extrinsic_camera_matrix_filenames_multiviewx = [
    'extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml',
    'extr_Camera4.xml', 'extr_Camera5.xml', 'extr_Camera6.xml'
]


class MultiviewX(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'MultiviewX'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [640, 1000]
        self.num_cam, self.num_frame = 6, 400
        self.indexing = 'xy'
        self.worldgrid2worldcoord_mat = np.array([[0, 0.025, 0], [0.025, 0, 0], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)]
        )

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 1000

    def get_worldgrid_from_worldcoord(self, world_coord):
        coord_x, coord_y = world_coord
        grid_x = coord_x * 40
        grid_y = coord_y * 40
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        coord_x = grid_x / 40
        coord_y = grid_y / 40
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(
            os.path.join(intrinsic_camera_path, intrinsic_camera_matrix_filenames_multiviewx[camera_i]),
            flags=cv2.FILE_STORAGE_READ
        )
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(
            os.path.join(extrinsic_camera_path, extrinsic_camera_matrix_filenames_multiviewx[camera_i]),
            flags=cv2.FILE_STORAGE_READ
        )
        rvec = fp_calibration.getNode('rvec').mat().squeeze()
        tvec = fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=T.ToTensor(), target_transform=T.ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=False):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape
        self.reducedgrid_shape = [int(x / self.grid_reduce) for x in self.worldgrid_shape]

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x_center = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                                single_pedestrian['views'][cam]['xmax']) / 2),
                                           self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x_center > 0 and y_foot > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x_center)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x_center)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        return imgs, map_gt.float(), imgs_gt, frame

    def __len__(self):
        return len(self.map_gt.keys())


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18', device=None):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape = dataset.img_shape
        self.reducedgrid_shape = dataset.reducedgrid_shape
        self.device = device if device is not None else torch.device('cpu')

        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(
            dataset.base.intrinsic_matrices,
            dataset.base.extrinsic_matrices,
            dataset.base.worldgrid2worldcoord_mat
        )
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        self.upsample_shape = [int(x / dataset.img_reduce) for x in self.img_shape]
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        proj_mats = [map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat
                     for cam in range(self.num_cam)]
        self.register_buffer('proj_mats', torch.stack([torch.from_numpy(p) for p in proj_mats], dim=0))

        if arch == 'vgg11':
            base = vgg11(weights=None).features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            self.base = base.to(self.device)
            out_channel = 512
        elif arch == 'resnet18':
            base = resnet18(weights=None, replace_stride_with_dilation=[False, True, True])
            self.base = nn.Sequential(*list(base.children())[:-2]).to(self.device)
            out_channel = 512
        else:
            raise ValueError('arch must be vgg11 or resnet18')

        self.img_classifier = nn.Sequential(
            nn.Conv2d(out_channel, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1, bias=False)
        ).to(self.device)
        self.map_classifier = nn.Sequential(
            nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)
        ).to(self.device)

    def forward(self, imgs):
        bsz, num_cam, _, _, _ = imgs.shape
        assert num_cam == self.num_cam
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_feature = self.base(imgs[:, cam].to(self.device))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature)
            imgs_result.append(img_res)

            proj_mat = self.proj_mats[cam].repeat([bsz, 1, 1]).float().to(self.device)
            world_feature = warp_perspective(img_feature, proj_mat, self.reducedgrid_shape)
            world_features.append(world_feature)

        coord_map = self.coord_map.repeat([bsz, 1, 1, 1]).to(self.device)
        world_features = torch.cat(world_features + [coord_map], dim=1)
        map_result = self.map_classifier(world_features)
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        return map_result, imgs_result

    @staticmethod
    def get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(len(intrinsic_matrices)):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices

    @staticmethod
    def create_coord_map(img_size, with_r=False):
        h, w, _ = img_size
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = torch.from_numpy(grid_x / (w - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (h - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, h, w])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def build_dataloaders(args):
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize])
    if args.dataset == 'wildtrack':
        data_path = args.data_root or os.path.expanduser('~/Data/Wildtrack')
        base = Wildtrack(data_path)
    elif args.dataset == 'multiviewx':
        data_path = args.data_root or os.path.expanduser('~/Data/MultiviewX')
        base = MultiviewX(data_path)
    else:
        raise ValueError('dataset must be wildtrack or multiviewx')

    train_set = frameDataset(base, train=True, transform=transform,
                             grid_reduce=args.grid_reduce, img_reduce=args.img_reduce,
                             train_ratio=args.train_ratio, force_download=args.force_download)
    test_set = frameDataset(base, train=False, transform=transform,
                            grid_reduce=args.grid_reduce, img_reduce=args.img_reduce,
                            train_ratio=args.train_ratio, force_download=args.force_download)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader


def compute_loss(map_res, imgs_res, map_gt, imgs_gt, criterion, map_kernel, img_kernel, alpha):
    loss = criterion(map_res, map_gt.to(map_res.device), map_kernel)
    if imgs_res is not None:
        per_view_loss = 0
        for img_res, img_gt in zip(imgs_res, imgs_gt):
            per_view_loss += criterion(img_res, img_gt.to(img_res.device), img_kernel)
        loss = loss + per_view_loss / max(len(imgs_gt), 1) * alpha
    return loss


def compute_precision_recall(map_res, map_gt, cls_thres):
    pred = (map_res > cls_thres).int().to(map_gt.device)
    true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
    false_positive = pred.sum().item() - true_positive
    false_negative = map_gt.sum().item() - true_positive
    precision = true_positive / (true_positive + false_positive + 1e-4)
    recall = true_positive / (true_positive + false_negative + 1e-4)
    return precision, recall


def train_one_epoch(epoch, model, data_loader, optimizer, criterion, cls_thres, alpha, scheduler=None):
    model.train()
    loss_meter = AverageMeter()
    prec_meter = AverageMeter()
    recall_meter = AverageMeter()
    for batch_idx, (data, map_gt, imgs_gt, _) in enumerate(data_loader):
        optimizer.zero_grad()
        map_res, imgs_res = model(data)
        loss = compute_loss(map_res, imgs_res, map_gt, imgs_gt,
                            criterion, data_loader.dataset.map_kernel,
                            data_loader.dataset.img_kernel, alpha)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        precision, recall = compute_precision_recall(map_res, map_gt, cls_thres)
        loss_meter.update(loss.item(), data.size(0))
        prec_meter.update(precision, data.size(0))
        recall_meter.update(recall, data.size(0))

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch} [{batch_idx + 1}/{len(data_loader)}] '
                  f'loss {loss_meter.avg:.6f} prec {prec_meter.avg * 100:.1f}% '
                  f'recall {recall_meter.avg * 100:.1f}%')

    return loss_meter.avg, prec_meter.avg * 100, recall_meter.avg * 100


def evaluate(model, data_loader, criterion, cls_thres, alpha):
    model.eval()
    loss_meter = AverageMeter()
    prec_meter = AverageMeter()
    recall_meter = AverageMeter()
    with torch.no_grad():
        for data, map_gt, imgs_gt, _ in data_loader:
            map_res, imgs_res = model(data)
            loss = compute_loss(map_res, imgs_res, map_gt, imgs_gt,
                                criterion, data_loader.dataset.map_kernel,
                                data_loader.dataset.img_kernel, alpha)
            precision, recall = compute_precision_recall(map_res, map_gt, cls_thres)
            loss_meter.update(loss.item(), data.size(0))
            prec_meter.update(precision, data.size(0))
            recall_meter.update(recall, data.size(0))
    print(f'Val loss {loss_meter.avg:.6f} '
          f'prec {prec_meter.avg * 100:.1f}% recall {recall_meter.avg * 100:.1f}%')
    return loss_meter.avg, prec_meter.avg * 100, recall_meter.avg * 100


def main():
    parser = argparse.ArgumentParser(description='MVDet single-file trainer')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack',
                        choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--grid_reduce', type=int, default=4)
    parser.add_argument('--img_reduce', type=int, default=4)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--force_download', action='store_true')
    parser.add_argument('--output', type=str, default='logs_single')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = build_dataloaders(args)
    model = PerspTransDetector(train_loader.dataset, arch=args.arch, device=device)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs
    )
    criterion = GaussianMSE().to(device)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = os.path.join(args.output, f'{args.dataset}_{args.arch}_{timestamp}')
    os.makedirs(logdir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(epoch, model, train_loader, optimizer, criterion,
                        args.cls_thres, args.alpha, scheduler)
        evaluate(model, test_loader, criterion, args.cls_thres, args.alpha)
        ckpt_path = os.path.join(logdir, 'mvdet_single.pth')
        torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    main()
