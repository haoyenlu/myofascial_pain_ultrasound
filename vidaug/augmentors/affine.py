# Adapted from https://github.com/okankop/vidaug
# Copyright (c) 2018 okankop. MIT License (see THIRD_PARTY_NOTICES.md).

"""
Augmenters that apply affine transformations.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * RandomRotate
    * RandomResize
    * RandomTranslate
    * RandomShear
"""

import numpy as np
import numbers
import random
import scipy
import skimage
import PIL
import cv2
from PIL import Image

class RandomRotate(object):
    """
    Rotate video randomly by a random angle within given boundsi.

    Args:
        degrees (sequence or int): Range of degrees to randomly
        select from. If degrees is a number instead of sequence
        like (min, max), the range of degrees, will be
        (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class RandomResize(object):
    """
    Resize video bysoomingin and out.

    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].

        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate

        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = random.uniform(1 - self.rate, 1 + self.rate)

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_h, new_w)
        if isinstance(clip[0], np.ndarray):
            return [scipy.misc.imresize(img, size=(new_h, new_w),interp=self.interpolation) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC


class RandomTranslate(object):
    """
      Shifting video in X and Y coordinates.

        Args:
            x (int) : Translate in x direction, selected
            randomly from [-x, +x] pixels.

            y (int) : Translate in y direction, selected
            randomly from [-y, +y] pixels.
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __call__(self, clip):
        x_move = random.randint(-self.x, +self.x)
        y_move = random.randint(-self.y, +self.y)

        if isinstance(clip[0], np.ndarray):
            rows, cols, ch = clip[0].shape
            transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
            return [cv2.warpAffine(img, transform_mat, (cols, rows)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, x_move, 0, 1, y_move)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))


class RandomShear(object):
    """
    Shearing video in X and Y directions.

    Args:
        x (int) : Shear in x direction, selected randomly from
        [-x, +x].

        y (int) : Shear in y direction, selected randomly from
        [-y, +y].
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, clip):
        x_shear = random.uniform(-self.x, self.x)
        y_shear = random.uniform(-self.y, self.y)

        if isinstance(clip[0], np.ndarray):
            rows, cols, ch = clip[0].shape
            transform_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
            return [cv2.warpAffine(img, transform_mat, (cols, rows)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transform(img.size, PIL.Image.AFFINE, (1, x_shear, 0, y_shear, 1, 0)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip[0])))


class RandomZoom(object):
    def __init__(self, scale_range=(0.8, 1.2)):
        """
        Args:
            scale_range: (min_zoom, max_zoom). <1 = zoom out, >1 = zoom in.
            p: probability of applying zoom.
        """
        self.scale_range = scale_range

    def __call__(self, frames):
        """
        Args:
            frames: list of PIL.Image objects (all same size).
        Returns:
            list of PIL.Image objects after random zoom (same size as input).
        """
        if len(frames) == 0:
            return frames

        # Assume all frames have the same size
        w, h = frames[0].size

        # Sample a single zoom factor for the whole video
        scale = random.uniform(*self.scale_range)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        def zoom_single_frame(img: Image.Image) -> Image.Image:
            # Resize (zoom)
            img_zoom = img.resize((new_w, new_h), resample=Image.BILINEAR)

            if scale >= 1.0:
                # Zoom in: center-crop back to (w, h)
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                right = left + w
                bottom = top + h
                return img_zoom.crop((left, top, right, bottom))
            else:
                # Zoom out: pad into a (w, h) canvas
                # Use black background; you can change this if needed.
                canvas = Image.new(img.mode, (w, h))
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                canvas.paste(img_zoom, (left, top))
                return canvas

        return [zoom_single_frame(f) for f in frames]