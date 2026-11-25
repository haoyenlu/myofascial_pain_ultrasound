from torch.utils.data import Dataset
import torch
import os
import copy
from PIL import Image
import numpy as np


from einops import rearrange
import vidaug.augmentors as va
from dataclasses import dataclass

import cv2
from PIL import Image

def video_loader(file_path):
    video = cv2.VideoCapture(file_path)
    frames = []
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale for ultrasound
        pil_img = Image.fromarray(frame , 'L') # convert to PIL Image
        frames.append(pil_img)
    video.release()
    return frames



def imgResize(img: Image, target_size):
    return img.resize((target_size,target_size), Image.BICUBIC)


def img2tensor(img: np.array, repeat_chn = True):
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0) # add channel dimension
    if repeat_chn:  
        img = np.repeat(img, repeats=3, axis=0) # turn 1 channel to 3 channel of gray scale image
    img = torch.from_numpy(img)
    img = img / 255.0 # normalize
    return img.float()


@dataclass
class VideoSample:
    label: int
    video: str
    frame_indices: list
    demo: dict





class VideoDataset(Dataset):
    '''Generate the dataset using the video'''
    def __init__(self, 
                 video_path, 
                 labels,
                 frame_size,
                 use_demo    = False,
                 sex_feature = None, 
                 age_feature = None,
                 ppt_feature = None,
                 augmentation = None,
                 channel_first = True,
                 repeat_chn = True,
                 samplers = [va.RandomFrameSampler,va.SlidingWindowSampler]):
        """
        Args:
            video_path: root dir of the videos
            spatial_transform: transformation for each frame
            temporal_transform: transformation for timestep
            sample_duration: number of frames for each data
        """
        
        assert len(video_path) == len(labels), "Number of Videos should be the same as the labels"

        self.video_path = video_path
        self.labels     = labels
        self.frame_size = frame_size
        
        self.sex_feature = sex_feature
        self.age_feature = age_feature  
        self.ppt_feature = ppt_feature

        self.repeat_chn  = repeat_chn
        self.use_demo    = use_demo

        self.samplers = samplers
        self.augmentation = augmentation
        self.channel_first = channel_first
        self.frame_cache = {}

        self.__prepare_data()

    def __prepare_data(self):
        self.data = []
        for idx, video in enumerate(self.video_path):
            label = self.labels[idx]
            video_frames = video_loader(video)
            total_frames = len(video_frames)
            for sampler in self.samplers:
                video_samples = sampler(total_frames)
                for s in video_samples:
                    item = VideoSample(video=video, 
                                       label=label, 
                                       frame_indices=s, 
                                       demo={
                                        'sex': self.sex_feature[idx],
                                        'age': self.age_feature[idx],
                                        'ppt': self.ppt_feature[idx]
                                       })
                    self.data.append(item)

        print(f"Total {len(self.data)} samples prepared from {len(self.video_path)} videos.")


    def _get_frames(self, video_path):
        if video_path not in self.frame_cache:
            self.frame_cache[video_path] = video_loader(video_path)
        return self.frame_cache[video_path]


    def _to_tensor(self, data):
        data = np.array(data)
        data = torch.from_numpy(data).float()
        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            frames: the target frame
            labels: the target frame label
        """
        video = self.data[index].video
        label = self.data[index].label
        frame_indices = self.data[index].frame_indices
        frames = self._get_frames(video) # load all the frame of the video
        clip = [frames[i] for i in frame_indices]   # get only the clip of the videos
        clip = self.augmentation(clip) if self.augmentation is not None else clip
        clip = [imgResize(img, target_size=self.frame_size) for img in clip]
        clip = np.array(clip)
        clip = img2tensor(clip, self.repeat_chn)
           
        
        patient = os.path.basename(video).split('_')[0]
        intervention = os.path.basename(video).split('_')[1]

        label = torch.Tensor((label,)).float()

        if self.use_demo:
            sex = torch.Tensor((self.data[index].demo['sex'],)).float()
            age = torch.Tensor((self.data[index].demo['age'],)).float()
            ppt = torch.Tensor(self.data[index].demo['ppt']).float()


        data = {"clip" : clip, 
                "label": label, 
                "patient":patient , 
                "intervention": intervention,
                'demo': {
                    'sex': sex,
                    'age': age,
                    'ppt': ppt
                } if self.use_demo else {}
        }

        return data
    

    def __len__(self):
        return len(self.data)
    


@dataclass
class VideoMAESample:
    video: str
    frame_indices: list


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 

class VideoMAEDataset(Dataset):
    def __init__(self, 
                video_path, 
                frame_size,
                num_frames: int = 12,
                patch_size: int = 16,
                augmentation = None,
                channel_first = True,
                mask_ratio: float = 0.7,
                samplers = [va.RandomFrameSampler,va.SlidingWindowSampler]):
        """
        Args:
            video_path: root dir of the videos
            spatial_transform: transformation for each frame
            temporal_transform: transformation for timestep
            sample_duration: number of frames for each data
        """
        
        
        self.video_path = video_path
        self.frame_size = frame_size
        self.mask_ratio = mask_ratio
        self.num_frames = num_frames
        self.patch_size = patch_size

        self.samplers = samplers
        self.augmentation = augmentation
        self.channel_first = channel_first

        window_size = (self.num_frames // 2, self.frame_size // self.patch_size, self.frame_size // self.patch_size)

        self.mask_generator = TubeMaskingGenerator(window_size, mask_ratio)
        self.__prepare_data()

    def __prepare_data(self):
        self.data = []
        for idx, video in enumerate(self.video_path):
            video_frames = video_loader(video)
            total_frames = len(video_frames)
            for sampler in self.samplers:
                video_samples = sampler(total_frames)
                for s in video_samples:
                    item = VideoMAESample(video=video, frame_indices=s)
                    self.data.append(item)

        print(f"Total {len(self.data)} samples prepared from {len(self.video_path)} videos.")



    def _to_tensor(self, data):
        data = np.array(data)
        data = torch.from_numpy(data).float()
        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            frames: the target frame
            labels: the target frame label
        """
        video = self.data[index].video
        frame_indices = self.data[index].frame_indices
        frames = video_loader(video) # load all the frame of the video
        clip = [frames[i] for i in frame_indices]   # get only the clip of the videos
        clip = self.augmentation(clip) if self.augmentation is not None else clip
        clip = [imgResize(img, target_size=self.frame_size) for img in clip]
        clip = np.array(clip)
        clip = img2tensor(clip)
        

        mask = self.mask_generator()
        patchify_clip = rearrange(clip, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=self.patch_size, p2=self.patch_size)
        _, C   = patchify_clip.shape
        label = patchify_clip[mask].reshape(-1,C) 
        
        patient = os.path.basename(video).split('_')[0]
        intervention = os.path.basename(video).split('_')[1]


        data = {"clip" : clip, 
                "mask": mask,
                "target": label,
                "patient":patient , 
                "intervention": intervention}

        return data
    

class VideoFrameDataset(Dataset):
    '''Flattened frame-level dataset for training a diffusion model on individual frames'''
    def __init__(self,
                 video_path,
                 frame_size,
                 frame_stride     = 1,      # skip frames to reduce near-duplicate redundancy
                 labels           = None,   # optional, only needed for conditional diffusion
                 augmentation     = None,
                 repeat_chn       = True,
                 normalize_range  = (-1, 1)):  # diffusion models expect [-1,1], not [0,1]

        self.video_path      = video_path
        self.frame_size      = frame_size
        self.frame_stride    = frame_stride
        self.labels          = labels
        self.augmentation    = augmentation
        self.repeat_chn      = repeat_chn
        self.normalize_range = normalize_range

        self._frame_cache = {}  # path -> list[PIL.Image], decoded once
        self.__prepare_data()

    def _get_frames(self, video_path):
        if video_path not in self._frame_cache:
            self._frame_cache[video_path] = video_loader(video_path)
        return self._frame_cache[video_path]

    def __prepare_data(self):
        self.data = []  # each entry: (video_path, frame_idx, demo_dict_or_None)
        for idx, video in enumerate(self.video_path):
            frames = self._get_frames(video)
            label  = self.labels[idx] if self.labels is not None else None
            for f_idx in range(0, len(frames), self.frame_stride):
                self.data.append((video, f_idx, label))

        print(f"Total {len(self.data)} frames prepared from {len(self.video_path)} videos.")

    def __getitem__(self, index):
        video, frame_idx, label = self.data[index]
        frames = self._get_frames(video)
        img    = frames[frame_idx]

        if self.augmentation is not None:
            img = self.augmentation(img) 

        img = imgResize(img, target_size=self.frame_size)
        img = np.array(img)
        img = img2tensor(img, self.repeat_chn)

        out = {"image": img}
        if label is not None:
            out["label"] = torch.tensor(label).float()

        return out

    def __len__(self):
        return len(self.data)