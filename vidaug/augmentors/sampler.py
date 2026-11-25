import random
from PIL import Image
import numpy as np

from abc import ABC


def pingpong_looping(clip_len, video_length):
    # pingpong looping
    cycle = np.concatenate([
        np.arange(video_length), np.arange(video_length-2, 0, -2)
    ])

    reps = (clip_len + len(cycle) - 1) // len(cycle)
    idx = np.tile(cycle, reps)[:clip_len]

    return [idx]


class Sampler(ABC):
    """Abstract base class for video samplers."""
    def __call__(self, video):
        raise NotImplementedError("Sampler subclasses must implement the __call__ method.")

class SlidingWindowSampler:
    """Samples clips from a video using a sliding window approach."""
    def __init__(self, clip_len, step):
        """
        Args:
            clip_len (int): Length of each clip to sample.
            step (int): Step size for the sliding window.
        """
        self.clip_len = clip_len
        self.step = step

    def __call__(self, video_length: int):
        # assert self.clip_len <= video_length, f"Clip Length {self.clip_len} is larger than the video length {video_length}"
        if self.clip_len > video_length:
            # pingpong looping
            cycle = np.concatenate([
                np.arange(video_length), np.arange(video_length-2, 0, -1)
            ])

            reps = (self.clip_len + len(cycle) - 1) // len(cycle)
            idx = np.tile(cycle, reps)[:self.clip_len]

            return [idx]
        
        clips = []
        for start in range(0, video_length - self.clip_len + 1, self.step):
            end = start + self.clip_len
            clips.append(list(range(start, end)))
        return clips


class RandomStrideSampler:
    """Randomly samples clips from a video with a random stride."""
    def __init__(self, clip_len, num_clips: int = 1):
        """
        Args:
            clip_len (int): Length of each clip to sample.
            max_stride (int): Maximum stride between frames in the clip.
        """
        self.num_clips = num_clips  
        self.clip_len = clip_len


    def __call__(self, video_length: int):
        clips = []
        max_stride_allowed = max(1, (video_length - 1) // (self.clip_len - 1))
        for _ in range(self.num_clips):
            stride = random.randint(1, max_stride_allowed)
            start = random.randint(0, video_length - stride * (self.clip_len - 1) - 1)
            clip = list(range(start, start + stride * self.clip_len, stride))
            clips.append(clip)
        return clips



class RandomFrameSampler:
    def __init__(self, num_frames, num_clips: int = 1):
        self.num_frames = num_frames
        self.num_clips = num_clips

    def __call__(self, video_length: int):
        # assert video_length >= self.num_frames, f"Video length must be greater than or equal to num_frames, got {video_length} and {self.num_frames}"

        if self.num_frames > video_length:
            return []
            # pingpong looping
            cycle = np.concatenate([
                np.arange(video_length), np.arange(video_length-2, 0, -2)
            ])

            reps = (self.clip_len + len(cycle) - 1) // len(cycle)
            idx = np.tile(cycle, reps)[:self.clip_len]

            return [idx]

        clips = []  
        for _ in range(self.num_clips):
            if video_length == self.num_frames:
                return [list(range(0,video_length))]
            else:
                frame_indices = sorted(random.sample(range(0,video_length), self.num_frames))
                clips.append(frame_indices)
        return clips