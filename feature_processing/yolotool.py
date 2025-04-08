'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\feature_processing\\yolotool.py
Author       : Yongzhao Chen
Date         : 2024-09-01 15:13:44
LastEditTime : 2024-09-01 15:42:18
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script provides utility functions and classes for image and video processing in YOLO pipelines. It includes:
1. `letterbox`: A function for resizing and padding images while maintaining aspect ratio.
2. `LoadImages`: A class for loading images and videos, supporting batch processing and transformations.

The script is designed to support YOLO-based feature extraction and inference tasks.
'''

import numpy as np
import cv2
import os
from pathlib import Path
import glob

# Supported image and video formats
IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)
VID_FORMATS = (
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints.

    Args:
        im (np.ndarray): Input image.
        new_shape (tuple): Target shape (height, width).
        color (tuple): Padding color. Defaults to (114, 114, 114).
        auto (bool): Whether to calculate padding automatically. Defaults to True.
        scaleFill (bool): Whether to stretch the image to fill the new shape. Defaults to False.
        scaleup (bool): Whether to allow scaling up. Defaults to True.
        stride (int): Stride for resizing. Defaults to 32.

    Returns:
        tuple: Resized and padded image, scaling ratio, and padding values.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding

    if auto:
        dw, dh = dw % stride, dh % stride  # adjust padding to be a multiple of stride
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into two sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, ratio, (dw, dh)

class LoadImages:
    """
    YOLOv5 image/video dataloader. Supports loading images and videos for inference.

    Args:
        path (str or list): Path to images/videos or a list of paths.
        img_size (int): Target image size for resizing. Defaults to 640.
        stride (int): Stride for resizing. Defaults to 32.
        auto (bool): Whether to calculate padding automatically. Defaults to True.
        transforms (callable): Optional transformations to apply to images.
        vid_stride (int): Frame stride for videos. Defaults to 1.
    """
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with paths
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # directory
            elif os.path.isfile(p):
                files.append(p)  # single file
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional transformations
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # initialize first video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # apply transformations
        else:
            im = letterbox(im0, new_shape=(self.img_size, self.img_size), stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # make contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """
        Initialize a new video.

        Args:
            path (str): Path to the video file.
        """
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame = 0
