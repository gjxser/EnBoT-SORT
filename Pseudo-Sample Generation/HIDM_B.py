import cv2
import numpy as np
import random
import os
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_motion_blur(image, degree, angle):
    kernel = np.zeros((degree, degree), dtype=np.float32)
    center = degree // 2
    for i in range(degree):
        kernel[center, i] = 1
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((center, center), angle, 1), (degree, degree))
    kernel = kernel / degree
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def add_noise(image, noise_type='gaussian', mean=0, stddev=0.04):
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_image = cv2.add(image.astype(np.float32), noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    elif noise_type == 'poisson':
        noise = np.random.poisson(image / 255.0 * stddev).astype(np.float32)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    return image

# === HIDM-B ===
def degrade_background_sequence1(background_images, num_frames_per_segment=100, max_alpha=0.6, max_beta=1):
    num_frames = len(background_images)
    degraded_images = []
    for i in range(0, num_frames, num_frames_per_segment):
        segment = background_images[i:i + num_frames_per_segment]
        if len(segment) == 0:
            continue
        alpha_start, alpha_end = random.uniform(0.2, max_alpha), random.uniform(1.0, max_alpha)
        beta_start, beta_end = random.uniform(0, -max_beta), random.uniform(0, max_beta)
        alphas = np.linspace(alpha_start, alpha_end, len(segment))
        betas = np.linspace(beta_start, beta_end, len(segment))
        for j, frame in enumerate(segment):
            degraded_frame = adjust_brightness_contrast(frame, alphas[j], betas[j])
            speed_factor = random.uniform(5, 15)
            angle = random.uniform(0, 360)
            degraded_frame = add_motion_blur(degraded_frame, int(speed_factor), angle)
            degraded_frame = add_noise(degraded_frame, noise_type='gaussian', mean=0, stddev=0.04)
            degraded_images.append(degraded_frame)
    return degraded_images

def random_anisotropic_gaussian_blur(img):
    kernel_width = 5 * 2 + 1
    kernel_height = 5 * 2 + 1
    sigma_x = 1.5
    sigma_y = 2.0
    return cv2.GaussianBlur(img, (kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
def random_motion_blur(img):
    kernel_size = random.randint(1, 7)
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    direction = random.choice(['horizontal', 'vertical'])
    if direction == 'horizontal':
        kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    else:
        kernel_motion_blur[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    return cv2.filter2D(img, -1, kernel_motion_blur)
def random_downsampling(img):
    """随机下采样图像"""
    # 根据图片大小设置下采样的范围
    scale = random.uniform(0.25, 1)  # 对应2x, 3x, 4x下采样
    interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    h, w = img.shape[:2]
    new_height = int(h * scale)
    new_width = int(w * scale)
    return cv2.resize(img, (new_width, new_height), interpolation=interpolation)

def random_upsampling(img, size):
    # 根据图片大小设置下采样的范围
    interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    return cv2.resize(img, dsize=size, interpolation=interpolation)

