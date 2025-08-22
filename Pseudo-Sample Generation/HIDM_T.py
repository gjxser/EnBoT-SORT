import cv2
import numpy as np
import random
import os
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def add_motion_blur(image, degree, angle):
    kernel = np.zeros((degree, degree), dtype=np.float32)
    center = degree // 2
    for i in range(degree):
        kernel[center, i] = 1
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((center, center), angle, 1), (degree, degree))
    kernel = kernel / degree
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def random_anisotropic_gaussian_blur(img):
    kernel_width = 3
    kernel_height = 3
    sigma_x = 0.5
    sigma_y = 0.8
    return cv2.GaussianBlur(img, (kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)

def rescale_target(target_image, scale):
    h, w = target_image.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(target_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# === HIDM-T ===
def degrade_target_sequence(target_images, num_frames, min_scale=0.2, max_scale=0.5):
    degraded_targets = []
    for target_image in target_images:
        h, w = target_image.shape
        scales = np.linspace(min_scale, max_scale, num_frames // 2).tolist()
        scales += scales[::-1]
        for scale in scales:
            scaled_target = rescale_target(target_image, scale)
            speed_factor = random.uniform(2, 6)
            angle = random.uniform(0, 360)
            #blurred_target=random_anisotropic_gaussian_blur(scaled_target)
            blurred_target = add_motion_blur(scaled_target, int(speed_factor), angle)
            degraded_targets.append(blurred_target)
    return degraded_targets

def random_motion_blur(img):
    kernel_size = random.randint(3, 7)
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    direction = random.choice(['horizontal', 'vertical'])
    if direction == 'horizontal':
        kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    else:
        kernel_motion_blur[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    return cv2.filter2D(img, -1, kernel_motion_blur)

def random_rotate(img, mask):
    angle = random.randint(-60, 60)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0])), cv2.warpAffine(mask, rotation_matrix, (img.shape[1], img.shape[0]))

def random_downsampling(img, mask):
    min_size = 0.5 if img.shape[0] < 50 else 0.25
    max_size = 1 if img.shape[0] < 50 else 0.5
    scale = random.uniform(min_size, max_size)  # 对应2x, 3x, 4x下采样
    interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    h, w = img.shape[:2]
    new_height = int(h * scale)
    new_width = int(w * scale)
    return cv2.resize(img, (new_width, new_height), interpolation=interpolation), cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

