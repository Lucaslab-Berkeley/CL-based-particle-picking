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
from . import filters
import torchvision.transforms as transforms
import warnings
from tqdm import tqdm
import time
from scipy.stats import gaussian_kde

import biotite.structure.io as strucio
import biotite.structure as struc
from Bio import PDB
from Bio.PDB import MMCIFParser
from Bio.PDB.vectors import Vector
import open3d as o3d

import mrcfile
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
import glob
import time
import mrcfile

from typing import Tuple

warnings.simplefilter('ignore', PDB.PDBExceptions.PDBConstructionWarning)

def load_template_data(DATA_DIR):
    all_filepath = []
    
    for cls in sorted(glob.glob(os.path.join(DATA_DIR, '*'))):
        pcs = sorted(glob.glob(os.path.join(cls, '*')))
        all_filepath += pcs
    # print(all_filepath)    
    return all_filepath

def load_img_data(DATA_DIR):
    all_filepath = []
    
    for cls in sorted(glob.glob(os.path.join(DATA_DIR, '*'))):
        imgs = sorted(glob.glob(os.path.join(cls, '*')))
        if len(imgs)%2 == 0:
            all_filepath += imgs
        else:
            all_filepath += imgs[:-1]
    # print(all_filepath)    
    return all_filepath

def load_struc_files_npy(fp):
    return np.load(fp) # downsampled numpy array

def load_img_file(fp):
    with mrcfile.open(fp) as mrc:
        data = mrc.data.copy()
        
    return data

class ScatteringPotential:
    """This class holds a 3D scattering potential array and provides methods for
    taking projections at different orientations via the Fourier slice operation.
    Code credits: Matthew Giammar
    
    Attributes:
        potential_array (np.ndarray): The 3D scattering potential array.
        potential_array_fft (np.ndarray): The FFT of the scattering potential.
        pixel_size (float): The size of the pixels in Angstroms.
        
        _interpolator (RegularGridInterpolator): A pre-computed interpolator for the
            spatial frequencies of the scattering potential.
        _slice_shape (Tuple[int, int]): The shape of the scattering potential in the
            x and y dimensions used to define the Fourier slice grid.
    
    Methods:
        from_mrc(mrc_path: str) -> ScatteringPotential: Create a ScatteringPotential
            object from an MRC file.
        take_fourier_slice(phi: float, theta: float, psi: float) -> np.ndarray: Takes a
            Fourier slice of the scattering potential at an orientation from the given
            Euler angles.
        take_projection(phi: float, theta: float, psi: float) -> np.ndarray: Take a
            real-space projection of the scattering potential at an orientation from the
            given Euler angles.
    """
    
    potential_array: np.ndarray
    potential_array_fft: np.ndarray
    pixel_size: float  # In Angstroms, assume cubic pixels
    
    _interpolator: RegularGridInterpolator
    _slice_shape: Tuple[int, int]
    
    @classmethod
    def from_mrc(cls, mrc_path: str):
        """Create a ScatteringPotential object from an MRC file."""
        with mrcfile.open(mrc_path) as mrc:
            potential_array = mrc.data.copy()
            pixel_size = mrc.voxel_size.x
            
        # Conform to cisTEM convention (reversed axes)
        potential_array = np.swapaxes(potential_array, 0, -1)
        
        return cls(potential_array, pixel_size)
    
    def __init__(self, potential_array: np.ndarray, pixel_size: float):
        self.potential_array = potential_array
        self.pixel_size = pixel_size
        
        # Precompute the FFT of the scattering potential
        # NOTE: The real-space potential is first fft-shifted to correct for the
        # odd-valued frequencies when taking a Fourier slice. See (TODO) for more info
        self.potential_array_fft = np.fft.fftshift(potential_array) # if we don't do this the phase will be shifted in an undesirable way
        self.potential_array_fft = np.fft.fftn(self.potential_array_fft)
        self.potential_array_fft = np.fft.fftshift(self.potential_array_fft)
        
        dim = [np.arange(s) - s // 2 for s in potential_array.shape]
        self._interpolator = RegularGridInterpolator(
            points=dim,
            values=self.potential_array_fft,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        self._slice_shape = potential_array.shape[:-1]  # (x, y) dimensions
        
    def take_fourier_slice(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Takes a Fourier slice of the pre-computed scattering potential at an
        orientation from the given Euler angles. The angles are in radians, and the
        rotation convention is ZYZ.
        
        Returned array is in the Fourier domain and centered (zero-frequency in center)
        
        Args:
            phi (float): The rotation around the Z axis in radians.
            theta (float): The rotation around the Y' axis in radians.
            psi (float): The rotation around the Z'' axis in radians.
            
        Returns:
            np.ndarray: The Fourier slice of the scattering potential.
        """
        rot = Rotation.from_euler("ZYZ", [phi, theta, psi]) # A convention used in most microscopic and robotic systems...
        
        # Generate a grid of integer coordinates at z = 0 then rotate
        x = np.arange(self._slice_shape[0]) - self._slice_shape[0] // 2
        y = np.arange(self._slice_shape[1]) - self._slice_shape[1] // 2
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        
        coordinates = np.stack([xx, yy, zz], axis=-1)
        coordinates = coordinates.reshape(-1, 3)
        coordinates = rot.apply(coordinates)
        
        # Interpolate the scattering potential at the rotated coordinates
        fourier_slice = self._interpolator(coordinates)
        fourier_slice = fourier_slice.reshape(xx.shape)
        
        return fourier_slice
    
    def take_projection(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Take a real-space projection of the scattering potential at an orientation
        from the given Euler angles. The angles are in radians, and the rotation
        convention is ZYZ.
        
        Returned array is in real-space.
        
        Args:
            phi (float): The rotation around the Z axis in radians.
            theta (float): The rotation around the Y' axis in radians.
            psi (float): The rotation around the Z'' axis in radians.
            
        Returns:
            np.ndarray: The projection of the scattering potential.
        """
        fourier_slice = self.take_fourier_slice(phi, theta, psi)
        
        fourier_slice = np.fft.ifftshift(fourier_slice)
        projection = np.fft.ifftn(fourier_slice)
        projection = np.fft.ifftshift(projection)
        
        return np.real(projection)

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
            
    def render_image(self, pc_path):
        fp = f"{'/'.join(pc_path.split('/')[:-4])}/volume/{'/'.join(pc_path.split('/')[-3:])[:-3]}mrc"
        density = ScatteringPotential.from_mrc(fp)

        phi = random.uniform(0, 2*np.pi)  # rotation around Z axis
        theta = random.uniform(0, np.pi) # rotation around Y axis
        psi = random.uniform(0, 2*np.pi)  # rotation around new Z axis

        projection = density.take_projection(phi, theta, psi)

        return projection
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
                
        label_map = {'ribosome':0, 'neg_class':1}
        
        if(item < self.num_cache):
            # read numpy from RAM
            pointcloud_1 = self.all_pcs[item]
            pointcloud_2 = self.all_pcs[item]
        else:
            pointcloud_1 = load_struc_files_npy(self.data[item])
            pointcloud_2 = load_struc_files_npy(self.data[item])

        # start = time.time()
        render_img = self.render_image(pcd_path).astype('float32') # time = around 6s
        # print(f"Processing time = {time.time() - start}s")
        render_img = self.transform(render_img) 

        point_t1 = self.trans_1(pointcloud_1)
        point_t2 = self.trans_2(pointcloud_2)

        pointcloud = (point_t1, point_t2)
    
        return pointcloud, render_img, label_map[pcd_path.split("/")[-2]] 
    
    def __len__(self):
        return len(self.data)
    
def prenorm(fp):
    if "_projection_" not in fp:
        img = load_img_file(fp)
        return (img - np.mean(img)) / np.std(img)
    else:
        return load_img_file(fp)
    
class cryoEM_cl2d2d_loader(Dataset):
    def __init__(self, fp = "", img_transform = None, type_="train", filter_type="lowpass", filter_cutoff=0.1, fourier_transformed=False):
        self.data = load_img_data(fp) # ribosome, background
        self.transform = img_transform
        self.type = type_
        self.filter_type = filter_type
        self.filter_cutoff = filter_cutoff
        self.fourier_transformed = fourier_transformed
    
    def __getitem__(self, item):                
        label_map = {'ribosome':0, 'background':1}
        
        img1_fp = self.data[2*item]
        img2_fp = self.data[2*item + 1]
        
        img1_class = img1_fp.split("/")[-2]
        img2_class = img2_fp.split("/")[-2]
                
        assert img1_class == img2_class, "Classes different in positive pair !!!"
        
        img1 = prenorm(img1_fp)
        img2 = prenorm(img2_fp)
        
        img1_filtered = filters.filter_image(img1, self.filter_cutoff, filter_type = self.filter_type)
        img2_filtered = filters.filter_image(img2, self.filter_cutoff, filter_type = self.filter_type)
        
        if self.fourier_transformed:
            img1_filtered = np.fft.fftshift(np.fft.fft2(img1_filtered))
            img2_filtered = np.fft.fftshift(np.fft.fft2(img2_filtered))

        img_t1 = self.transform(img1_filtered.astype('float32'))
        img_t2 = self.transform(img2_filtered.astype('float32'))
    
        return img_t1, img_t2, label_map[img1_class] 
    
    def __len__(self):
        return len(self.data)//2