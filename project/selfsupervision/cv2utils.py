import numpy as np
import os
import torch

import torchvision.transforms as trn

import itertools
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
import math

randomly_crop = trn.RandomCrop(32, padding=4)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _get_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute matrix for affine transformation
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]

    angle = math.radians(angle)
    shear = math.radians(shear)
    # scale = 1.0 / scale

    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
    RSS = np.array(
        [
            [math.cos(angle) * scale, -math.sin(angle + shear) * scale, 0],
            [math.sin(angle) * scale, math.cos(angle + shear) * scale, 0],
            [0, 0, 1],
        ]
    )
    matrix = T @ C @ RSS @ np.linalg.inv(C)

    return matrix[:2, :]


def affine(
    img,
    angle,
    translate,
    scale,
    shear,
    interpolation=cv2.INTER_CUBIC,
    mode=cv2.BORDER_CONSTANT,
    fillcolor=0,
):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (numpy ndarray): numpy ndarray to be transformed.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        interpolation (``cv2.INTER_NEAREST` or ``cv2.INTER_LINEAR`` or ``cv2.INTER_AREA``, ``cv2.INTER_CUBIC``):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, it is set to ``cv2.INTER_CUBIC``, for bicubic interpolation.
        mode (``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE`` or ``cv2.BORDER_REFLECT`` or ``cv2.BORDER_REFLECT_101``)
            Method for filling in border regions.
            Defaults to cv2.BORDER_CONSTANT, meaning areas outside the image are filled with a value (val, default 0)
        val (int): Optional fill color for the area outside the transform in the output image. Default: 0
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be numpy Image. Got {}".format(type(img)))

    assert (
        isinstance(translate, (tuple, list)) and len(translate) == 2
    ), "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.shape[0:2]
    center = (img.shape[1] * 0.5 + 0.5, img.shape[0] * 0.5 + 0.5)
    matrix = _get_affine_matrix(center, angle, translate, scale, shear)

    if img.shape[2] == 1:
        return cv2.warpAffine(
            img,
            matrix,
            output_size[::-1],
            interpolation,
            borderMode=mode,
            borderValue=fillcolor,
        )[:, :, np.newaxis]
    else:
        return cv2.warpAffine(
            img,
            matrix,
            output_size[::-1],
            interpolation,
            borderMode=mode,
            borderValue=fillcolor,
        )
