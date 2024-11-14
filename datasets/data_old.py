import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image
from .plyfile import load_ply
from . import data_utils as d_utils
import torchvision.transforms as transforms
import warnings
from tqdm import tqdm
import time
from scipy.stats import gaussian_kde

import biotite.structure.io as strucio
import biotite.structure as struc
from scipy.spatial.transform import Rotation
from Bio import PDB
from Bio.PDB import MMCIFParser
from Bio.PDB.vectors import Vector
import open3d as o3d
import time

from PIL import ImageFile

warnings.simplefilter('ignore', PDB.PDBExceptions.PDBConstructionWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_DIR = 'data/ShapeNetRendering'

# trans_1 = transforms.Compose(
#             [
#                 d_utils.PointcloudToTensor(),
#                 d_utils.PointcloudNormalize(),
#             ])

# trans_2 = transforms.Compose(
#             [
#                 d_utils.PointcloudToTensor(),
#                 d_utils.PointcloudNormalize(),
#             ])

shapenet_trans_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])
    
shapenet_trans_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

def load_modelnet_data(partition):
    # BASE_DIR = ''
    DATA_DIR = "/hpc/projects/group.czii/kithmini.herath/crosspoint-data" # os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_ScanObjectNN(partition):
    BASE_DIR = '/hpc/projects/group.czii/kithmini.herath/crosspoint-data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label

def load_shapenet_data():
    # BASE_DIR = ''
    DATA_DIR = "/hpc/projects/group.czii/kithmini.herath/crosspoint-data" # os.path.join(BASE_DIR, 'data')
    all_filepath = []
    
    # print(os.path.join(DATA_DIR, 'ShapeNet/*'))

    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs
    # print(all_filepath)    
    return all_filepath

def get_render_imgs(pcd_path):
    path_lst = pcd_path.split('/')
    # print(path_lst)
    path_lst[-3] = 'ShapeNetRendering'
    # print(path_lst)
    path_lst[-1] = path_lst[-1][:-4]
    # print(path_lst)
    path_lst.append('rendering')
    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))
    
    return img_path_list

class ShapeNetRender(Dataset):
    def __init__(self, img_transform = None, n_imgs = 1):
        self.data = load_shapenet_data()
        self.transform = img_transform
        self.n_imgs = n_imgs
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
        render_img_path = random.choice(get_render_imgs(pcd_path))
        # render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)
        # render_img_list = []
        # for render_img_path in render_img_path_list:
        render_img = Image.open(render_img_path).convert('RGB')
        render_img = self.transform(render_img)  #.permute(1, 2, 0)
            # render_img_list.append(render_img)
        pointcloud_1 = load_ply(self.data[item])
        # pointcloud_orig = pointcloud_1.copy()
        pointcloud_2 = load_ply(self.data[item])
        point_t1 = shapenet_trans_1(pointcloud_1)
        point_t2 = shapenet_trans_2(pointcloud_2)

        # pointcloud = (pointcloud_orig, point_t1, point_t2)
        pointcloud = (point_t1, point_t2)
        return pointcloud, render_img # render_img_list

    def __len__(self):
        return len(self.data)

# class ShapeNetRender(Dataset):
#     def __init__(self, img_transform = None, n_imgs = 1):
#         self.data = load_shapenet_data()
#         self.transform = img_transform
#         self.n_imgs = n_imgs
    
#     def __getitem__(self, item):
#         pcd_path = self.data[item]
#         render_img_path = random.choice(get_render_imgs(pcd_path))
#         # render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)
#         # render_img_list = []
#         # for render_img_path in render_img_path_list:
#         render_img = Image.open(render_img_path).convert('RGB') # Image originally opens as RGBA on PIL.Image type
#         render_img = self.transform(render_img)  #.permute(1, 2, 0)
#             # render_img_list.append(render_img)
#         pointcloud_1 = load_ply(self.data[item])
#         # pointcloud_orig = pointcloud_1.copy()
#         pointcloud_2 = load_ply(self.data[item])
#         point_t1 = pointcloud_1 #trans_1(pointcloud_1)
#         point_t2 = pointcloud_2 #trans_2(pointcloud_2)

#         # pointcloud = (pointcloud_orig, point_t1, point_t2)
#         pointcloud = (point_t1, point_t2)
#         return pointcloud, render_img # render_img_list

#     def __len__(self):
#         return len(self.data)
    
########################################################################################################################
    
def load_template_data(DATA_DIR):

    all_filepath = []
    
    for cls in sorted(glob.glob(os.path.join(DATA_DIR, '3d_models_ds/*'))):
        pcs = sorted(glob.glob(os.path.join(cls, '*')))
        all_filepath += pcs
    # print(all_filepath)    
    return all_filepath

def load_template_data_2classes(DATA_DIR):

    all_filepath = []
    
    for cls in sorted(glob.glob(os.path.join(DATA_DIR, '3d_models_ds/*'))):
        if cls.split('/')[-1] in ['ribosome', 'proteasome']:
            pcs = sorted(glob.glob(os.path.join(cls, '*')))
            all_filepath += pcs
    # print(all_filepath)    
    return all_filepath

def mmcif_to_coords(fp):
    # Parse the mmCIF file
    parser = MMCIFParser()
    structure = parser.get_structure("structure", fp)
    
    # Initialize a list to store coordinates
    coords = []
    
    # Iterate through the structure and extract coordinates
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.coord)
    
    # Convert the list of coordinates to a NumPy array
    coords_array = np.array(coords)
    
    return coords_array

def pdb_to_coords(fp):
    pdb_array = strucio.load_structure(fp)
    coords_array = np.zeros((len(pdb_array),3))

    for ii in range(len(pdb_array)):
        # Pull the Coordinates
        coords_array[ii] = pdb_array[ii].coord
        
    return coords_array

def get_projection_imgs(pcd_path):
    path_lst = pcd_path.split('/')
    # print(path_lst)
    path_lst[-3] = '2d_images'
    # print(path_lst)
    path_lst[-1] = path_lst[-1][:-4]
    
    DIR = '/'.join(path_lst)
    img_path_list = sorted(glob.glob(os.path.join(DIR, '*.npy')))
    
    return img_path_list

def rotate_point_cloud(points, angles):
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    return rotation.apply(points)

def orthographic_projection(points, axis):
    return np.delete(points, axis, 1)

def precompute_density(point_cloud):
    xy = point_cloud[:, :2]
    kde = gaussian_kde(xy.T)
    density = kde(xy.T)
    norm_density = (density - density.min()) / (density.max() - density.min())
    return 1 - norm_density

def downsample_fps(points):
    # Assuming your point cloud is stored in a numpy array called 'points'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform farthest point downsampling
    downsampled_pcd = pcd.farthest_point_down_sample(2048)

    # Get the downsampled points
    downsampled_points = np.asarray(downsampled_pcd.points)
    
    return downsampled_points

def load_struc_files(fp):
    if fp.split("/")[-1][-4:] == ".pdb":
        coordinates = pdb_to_coords(fp)
    elif fp.split("/")[-1][-4:] == ".cif":
        coordinates = mmcif_to_coords(fp)
        
    ds_coordinates = downsample_fps(coordinates)
    
    return ds_coordinates

def load_struc_files_npy(fp):
    return np.load(fp) # downsampled numpy array

# class cryoEM_loader(Dataset):
#     def __init__(self, fp = "", img_transform = None, n_imgs = 1, load_perc=1.0, type_="train"):
#         self.data = load_template_data(fp)
#         self.transform = img_transform
#         self.n_imgs = n_imgs
#         self.type = type_
        
#         self.num_cache = int(len(self.data)*load_perc)
        
#         self.all_pcs = []
        
#         warnings.warn(f"Loading {self.num_cache} point cloud files of the {self.type} set to RAM. This may take a while.")
        
#         for dp in tqdm(range(0, self.num_cache), desc="Loading data", unit="file"):
#             self.all_pcs.append(load_struc_files(self.data[dp]))
    
#     def __getitem__(self, item):
#         pcd_path = self.data[item]
#         render_img_path = random.choice(get_projection_imgs(pcd_path))
#         label_map = {'ribosome':0, 'atpase':1, 'proteasome':2}

#         render_img = Image.open(render_img_path) 
#         render_img = self.transform(render_img) 
        
#         # if self.type == 'train' or self.type == "val":
#         if(item < self.num_cache):
#             # read numpy from RAM
#             pointcloud_1 = self.all_pcs[item]
#             pointcloud_2 = self.all_pcs[item]
#         else:
#             pointcloud_1 = load_struc_files(self.data[item])
#             pointcloud_2 = load_struc_files(self.data[item])

#         # pointcloud_2 = pointcloud_1.copy()
#         point_t1 = trans_1(pointcloud_1)
#         point_t2 = trans_2(pointcloud_2)

#         pointcloud = (point_t1, point_t2)

#         return pointcloud, render_img, label_map[pcd_path.split("/")[-2]] 
#         # elif self.type == "test":
#         #     # need to return labels and images only since this is for classification during testing
#         #     return render_img, label_map[pcd_path.split("/")[-2]]
    
#     def __len__(self):
#         return len(self.data)
    
class cryoEM_npy_loader(Dataset):
    def __init__(self, fp = "", img_transform = None, pc1_transform = None, pc2_transform = None, n_imgs = 1, load_perc=1.0, type_="train"):
        self.data = load_template_data(fp)
        self.transform = img_transform
        self.trans_1 = pc1_transform
        self.trans_2 = pc2_transform
        self.n_imgs = n_imgs
        self.type = type_
        
        self.used_indices = set()  # Initialize the set of used indices -- remove when you have a completed dataset
        
        self.num_cache = int(len(self.data)*load_perc)
        
        self.all_pcs = []
        
        warnings.warn(f"Loading {self.num_cache} point cloud files of the {self.type} set to RAM. This may take a while.")
        
        for dp in tqdm(range(0, self.num_cache), desc="Loading data", unit="file"):
            self.all_pcs.append(load_struc_files_npy(self.data[dp]))
            
    def check_projection_imgs(self, pcd_path): # remove once you have a completed dataset
        projection_imgs = get_projection_imgs(pcd_path)
        return len(projection_imgs) != 24

    def find_alternative_pcd_path(self, current_index): # remove once you have a completed dataset
        for index in range(len(self.data)):
            if index != current_index and index not in self.used_indices and not self.check_projection_imgs(self.data[index]):
                return index
        return None
    
    def reset_used_indices(self): # remove once you have a completed dataset
        self.used_indices.clear()
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
        
        if self.check_projection_imgs(pcd_path):
            print(f"Incomplete projection directory: {pcd_path.split('/')[-1]}")
            alternative_index = self.find_alternative_pcd_path(item)
            if alternative_index is not None:
                item = alternative_index
                pcd_path = self.data[item]
                
        render_img_path = random.choice(get_projection_imgs(pcd_path))
        label_map = {'ribosome':0, 'atpase':1, 'proteasome':2}

        # render_img = Image.open(render_img_path) 
        render_img = np.load(render_img_path).astype('float32')
        render_img = self.transform(render_img) 
        
        # if self.type == 'train' or self.type == "val":
        if(item < self.num_cache):
            # read numpy from RAM
            pointcloud_1 = self.all_pcs[item]
            pointcloud_2 = self.all_pcs[item]
        else:
            pointcloud_1 = load_struc_files_npy(self.data[item])
            pointcloud_2 = load_struc_files_npy(self.data[item])

        # pointcloud_2 = pointcloud_1.copy()
        point_t1 = self.trans_1(pointcloud_1)
        point_t2 = self.trans_2(pointcloud_2)

        pointcloud = (point_t1, point_t2)
        
        self.used_indices.add(item) # Remove once you have a completed dataset

        return pointcloud, render_img, label_map[pcd_path.split("/")[-2]] 
        # elif self.type == "test":
        #     # need to return labels and images only since this is for classification during testing
        #     return render_img, label_map[pcd_path.split("/")[-2]]
    
    def __len__(self):
        return len(self.data)
    
class cryoEM_onTheFly_loader(Dataset):
    def __init__(self, fp = "", img_transform = None, pc1_transform = None, pc2_transform = None, n_imgs = 1, load_perc=1.0, type_="train"):
        self.data = load_template_data(fp)
        self.transform = img_transform
        self.trans_1 = pc1_transform
        self.trans_2 = pc2_transform
        self.n_imgs = n_imgs
        self.type = type_
        
        self.num_cache = int(len(self.data)*load_perc)
        
        self.all_pcs = []
        
        warnings.warn(f"Loading {self.num_cache} point cloud files of the {self.type} set to RAM. This may take a while.")
        
        for dp in tqdm(range(0, self.num_cache), desc="Loading data", unit="file"):
            self.all_pcs.append(load_struc_files_npy(self.data[dp]))
            
    def render_image(self, point_cloud):
        viewpoint = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        centroid = np.mean(point_cloud, axis=0)
        centered_pc = point_cloud - centroid

        rotated_cloud = rotate_point_cloud(centered_pc, viewpoint)
        projection = orthographic_projection(rotated_cloud, axis=2)  # Project onto XY plane

        norm_density = precompute_density(rotated_cloud)

        # Create a 300x300 numpy array filled with white
        img_array = np.ones((300, 300))

        # Normalize the projection coordinates to fit within 300x300 frame
        proj_min = projection.min(axis=0)
        proj_max = projection.max(axis=0)
        normalized_proj = (projection - proj_min) / (proj_max - proj_min) * 299
        normalized_proj = normalized_proj.astype(int)

        # Ensure the coordinates are within bounds
        x_coords = normalized_proj[:, 0]
        y_coords = normalized_proj[:, 1]
        valid_indices = (0 <= x_coords) & (x_coords < 300) & (0 <= y_coords) & (y_coords < 300)
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]

        norm_density_valid = norm_density[valid_indices]

        img_array[y_coords, x_coords] = norm_density_valid

        return img_array
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
                
        label_map = {'ribosome':0, 'atpase':1, 'proteasome':2}
        
        if(item < self.num_cache):
            # read numpy from RAM
            pointcloud_1 = self.all_pcs[item]
            pointcloud_2 = self.all_pcs[item]
        else:
            pointcloud_1 = load_struc_files_npy(self.data[item])
            pointcloud_2 = load_struc_files_npy(self.data[item])

        start = time.time()
        render_img = self.render_image(pointcloud_1).astype('float32')
        print(f"Processing time = {time.time() - start}s")
        render_img = self.transform(render_img) 

        point_t1 = self.trans_1(pointcloud_1)
        point_t2 = self.trans_2(pointcloud_2)

        pointcloud = (point_t1, point_t2)
    
        return pointcloud, render_img, label_map[pcd_path.split("/")[-2]] 
    
    def __len__(self):
        return len(self.data)
    
class cryoEM_npy_2classes_loader(Dataset):
    def __init__(self, fp = "", img_transform = None, pc1_transform = None, pc2_transform = None, n_imgs = 1, load_perc=1.0, type_="train"):
        self.data = load_template_data_2classes(fp)
        self.transform = img_transform
        self.trans_1 = pc1_transform
        self.trans_2 = pc2_transform
        self.n_imgs = n_imgs
        self.type = type_
        
        self.used_indices = set()  # Initialize the set of used indices -- remove when you have a completed dataset
        
        self.num_cache = int(len(self.data)*load_perc)
        
        self.all_pcs = []
        
        warnings.warn(f"Loading {self.num_cache} point cloud files of the {self.type} set to RAM. This may take a while.")
        
        for dp in tqdm(range(0, self.num_cache), desc="Loading data", unit="file"):
            self.all_pcs.append(load_struc_files_npy(self.data[dp]))
            
    def check_projection_imgs(self, pcd_path): # remove once you have a completed dataset
        projection_imgs = get_projection_imgs(pcd_path)
        return len(projection_imgs) != 24

    def find_alternative_pcd_path(self, current_index): # remove once you have a completed dataset
        for index in range(len(self.data)):
            if index != current_index and index not in self.used_indices and not self.check_projection_imgs(self.data[index]):
                return index
        return None
    
    def reset_used_indices(self): # remove once you have a completed dataset
        self.used_indices.clear()
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
        
        if self.check_projection_imgs(pcd_path):
            print(f"Incomplete projection directory: {pcd_path.split('/')[-1]}")
            alternative_index = self.find_alternative_pcd_path(item)
            if alternative_index is not None:
                item = alternative_index
                pcd_path = self.data[item]
                
        render_img_path = random.choice(get_projection_imgs(pcd_path))
        label_map = {'ribosome':0, 'proteasome':1}

        # render_img = Image.open(render_img_path) 
        render_img = np.load(render_img_path).astype('float32')
        render_img = self.transform(render_img) 
        
        # if self.type == 'train' or self.type == "val":
        if(item < self.num_cache):
            # read numpy from RAM
            pointcloud_1 = self.all_pcs[item]
            pointcloud_2 = self.all_pcs[item]
        else:
            pointcloud_1 = load_struc_files_npy(self.data[item])
            pointcloud_2 = load_struc_files_npy(self.data[item])

        # pointcloud_2 = pointcloud_1.copy()
        point_t1 = self.trans_1(pointcloud_1)
        point_t2 = self.trans_2(pointcloud_2)

        pointcloud = (point_t1, point_t2)
        
        self.used_indices.add(item) # Remove once you have a completed dataset

        return pointcloud, render_img, label_map[pcd_path.split("/")[-2]] 
        # elif self.type == "test":
        #     # need to return labels and images only since this is for classification during testing
        #     return render_img, label_map[pcd_path.split("/")[-2]]
    
    def __len__(self):
        return len(self.data)
    
class cryoEM_onTheFly_2classes_loader(Dataset):
    def __init__(self, fp = "", img_transform = None, pc1_transform = None, pc2_transform = None, n_imgs = 1, load_perc=1.0, type_="train"):
        self.data = load_template_data_2classes(fp)
        self.transform = img_transform
        self.trans_1 = pc1_transform
        self.trans_2 = pc2_transform
        self.n_imgs = n_imgs
        self.type = type_
        
        self.num_cache = int(len(self.data)*load_perc)
        
        self.all_pcs = []
        
        warnings.warn(f"Loading {self.num_cache} point cloud files of the {self.type} set to RAM. This may take a while.")
        
        for dp in tqdm(range(0, self.num_cache), desc="Loading data", unit="file"):
            self.all_pcs.append(load_struc_files_npy(self.data[dp]))
            
    def render_image(self, point_cloud):
        viewpoint = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        centroid = np.mean(point_cloud, axis=0)
        centered_pc = point_cloud - centroid

        rotated_cloud = rotate_point_cloud(centered_pc, viewpoint)
        projection = orthographic_projection(rotated_cloud, axis=2)  # Project onto XY plane

        norm_density = precompute_density(rotated_cloud)

        # Create a 300x300 numpy array filled with white
        img_array = np.ones((300, 300))

        # Normalize the projection coordinates to fit within 300x300 frame
        proj_min = projection.min(axis=0)
        proj_max = projection.max(axis=0)
        normalized_proj = (projection - proj_min) / (proj_max - proj_min) * 299
        normalized_proj = normalized_proj.astype(int)

        # Ensure the coordinates are within bounds
        x_coords = normalized_proj[:, 0]
        y_coords = normalized_proj[:, 1]
        valid_indices = (0 <= x_coords) & (x_coords < 300) & (0 <= y_coords) & (y_coords < 300)
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]

        norm_density_valid = norm_density[valid_indices]

        img_array[y_coords, x_coords] = norm_density_valid

        return img_array
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
                
        label_map = {'ribosome':0, 'proteasome':1}
        
        if(item < self.num_cache):
            # read numpy from RAM
            pointcloud_1 = self.all_pcs[item]
            pointcloud_2 = self.all_pcs[item]
        else:
            pointcloud_1 = load_struc_files_npy(self.data[item])
            pointcloud_2 = load_struc_files_npy(self.data[item])

        render_img = self.render_image(pointcloud_1).astype('float32')
        render_img = self.transform(render_img) 

        point_t1 = self.trans_1(pointcloud_1)
        point_t2 = self.trans_2(pointcloud_2)

        pointcloud = (point_t1, point_t2)
    
        return pointcloud, render_img, label_map[pcd_path.split("/")[-2]] 
    
    def __len__(self):
        return len(self.data)

    
########################################################################################################################
    
class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
        
        

