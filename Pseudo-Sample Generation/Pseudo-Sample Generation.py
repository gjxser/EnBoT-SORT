
import cv2
import numpy as np
import random
import os
import torch
from scipy.cluster.hierarchy import weighted
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from HIDM_B import degrade_background_sequence1
from HIDM_T import degrade_target_sequence
from tqdm import tqdm

#Trajectory Vision
def visualize_multiple_trajectories(trajectories, title='Trajectories'):
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], '-o', color=colors[i],  linewidth=0.01,markersize=2)
    plt.xlabel('W')
    plt.ylabel('H')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig("images_tracker/IRC_B/test/0/signal/trajectories/Figure_0.png",dpi=600,bbox_inches='tight')
    plt.show()

#Trajectory Generation
def generate_trajectory(start_point, num_frames, segment_length, image_size, target_size):
    trajectory = []
    x, y = start_point
    img_w, img_h = image_size
    target_w, target_h = target_size
    for _ in range(0, num_frames, segment_length):
        segment_type = random.choices(['linear', 'quadratic'],weights=[0.2,0.8])[0]
        if segment_type == 'linear':
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            for i in range(segment_length):
                new_x = x + i * dx
                new_y = y + i * dy
                # dynamic boundary detection
                new_x = max(0, min(new_x, img_w - target_w))
                new_y = max(0, min(new_y, img_h - target_h))

                trajectory.append((new_x, new_y))
        else:
            a = random.uniform(-0.02, 0.02) #0.005
            b = random.uniform(-5, 5)
            c = random.uniform(-5, 5)
            for i in range(segment_length):
                new_x = x + i
                new_y = y + a * i ** 2 + b * i + c

                # dynamic boundary detection
                #new_x = max(0, min(new_x, img_w - target_w))
                #new_y = max(0, min(new_y, img_h - target_h))

                trajectory.append((new_x, new_y))
        x, y = trajectory[-1]
    return np.array(trajectory[:num_frames])

def back_var(image):
    gray_image = image
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray_image, cv2.CV_64F).std()
    return variance

def target_var(image):
    variance = np.std(image)
    return variance

# Target Embedding and Fusion
def embed_multiple_targets(background_images, target_images, num_targets):
    h, w = background_images[0].shape
    num_frames = len(background_images)
    segment_length = 100
    image_size = (w, h)

    selected_targets = random.sample(target_images, num_targets)
    trajectories = []
    results = []

    for i in range(num_frames):
        frame = background_images[i]
        mask = np.zeros_like(frame)
        labels = []

        for target_idx, target_image in enumerate(selected_targets):
            th, tw = target_image.shape
            x_start, y_start = random.randint(0, w - tw - 1), random.randint(0, h - th - 1)

            '''
            b_var = back_var(frame[y_start:y_start + th, x_start:x_start + tw])
            t_var = target_var(target_image)
            #print(b_var,t_var)
            if b_var < t_var:
                brightness_factor = 1.2  
                contrast_factor = 1.2 
                target_image = target_image.astype(np.float32) 
                target_image = target_image * contrast_factor + (brightness_factor * 255 - 255) / 2
                target_image = np.clip(target_image, 0, 255).astype(np.uint8)
            else:
                target_image = np.clip(target_image, 0, 255).astype(np.uint8)
            '''

            #Trajectory Generation
            trajectory = generate_trajectory((x_start, y_start), num_frames, segment_length, image_size, (tw, th))
            if len(trajectories) <= target_idx:
                trajectories.append(trajectory)
            x, y = trajectories[target_idx][i].astype(int)

            '''
            for u in range(th):
                for v in range(tw):
                    if 0 <= x + u < w and 0 <= y + v < h:
                        T_uv = target_image[v, u]
                        if T_uv > 0:
                            I_uv = frame[y + v, x + u]
                            T_prime_uv = T_uv / np.max(target_image)
                            frame[y + v, x + u] = T_prime_uv * T_uv + (1 - T_prime_uv) * I_uv
                            mask[y + v, x + u] = 255
            '''
            for u in range(th):
                for v in range(tw):
                    if 0 <= x + u < w and 0 <= y + v < h:
                        I_uv = frame[y + v, x + u]
                        T_uv = target_image[v, u]
                        T_prime_uv = T_uv / np.max(target_image)
                        frame[y + v, x + u] = T_prime_uv * T_uv + (1 - T_prime_uv) * I_uv
                        mask[y + v, x + u] = 255

            if i==0:
                center_x=x+tw/2
                center_y = y + tw / 2
                with open(os.path.join(first_label_path,f"{idx}.txt"),'a') as f:
                    f.write(f"1,{target_idx+1},{int(center_x)},{int(center_y)},{tw},{th},1,1,1.0\n")

            # YOLO
            center_x = (x + tw / 2) / w
            center_y = (y + th / 2) / h
            norm_width = tw / w
            norm_height = th / h
            labels.append([0, center_x, center_y, norm_width, norm_height])

        '''
                while len(labels) < 5:
            labels.append([-1, -1, -1, -1, 0])
        labels = labels[:5]
        '''
        label_path = os.path.join(label_output_path, f"{idx}_{i}.txt")
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in label) + '\n')

        image_tensor = ToTensor()(frame)
        mask_tensor = ToTensor()(mask)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        results.append((image_tensor, mask_tensor, labels_tensor))
        cv2.imwrite(os.path.join(image_output_path, f"{idx}_{i}.jpg"), frame)
        cv2.imwrite(os.path.join(mask_output_path, f"{idx}_{i}.jpg"), mask)

    visualize_multiple_trajectories(trajectories, title='Target Trajectories')
    return results

# main fuction
def generate_degraded_dataset(background_images, target_images, num_targets, num_frames_per_segment=100):
    degraded_backgrounds = degrade_background_sequence1(background_images, num_frames_per_segment)
    degraded_targets = degrade_target_sequence(target_images, len(degraded_backgrounds))
    results = embed_multiple_targets(degraded_backgrounds, degraded_targets, num_targets)
    return results

total_folers = 0
pro = tqdm(range(0,total_folers+1),desc="process",unit="files")

for idx in pro:
    pro.set_description(f"process{idx}")
    '''
    background_path = f'/home/sys120/gjx/IRT_B/test/{idx}'
    target_path = 'target_images_ir'
    image_output_path = f'images_tracker/IRT_B/test/{idx}/mulit/image'
    mask_output_path = f'images_tracker/IRT_B/test/{idx}/mulit/mask'
    label_output_path = f'images_tracker/IRT_B/test/{idx}/mulit/label'
    first_label_path = f"images_tracker/IRT_B/test/{idx}/mulit/label_first"
    trajectories_path = f"images_tracker/IRT_B/test/{idx}/mulit/trajectories"
    '''
    background_path = f'C:/Users/guoji/Desktop/IRC_B/video{idx}/VI'
    target_path = 'target_images_ir'
    image_output_path = f'images_tracker/IRC_B/test/{idx}/mulit/image'
    mask_output_path = f'images_tracker/IRC_B/test/{idx}/mulit/mask'
    label_output_path = f'images_tracker/IRC_B/test/{idx}/mulit/label'
    first_label_path = f"images_tracker/IRC_B/test/{idx}/mulit/label_first"
    trajectories_path = f"images_tracker/IRC_B/test/{idx}/mulit/trajectories"

    os.makedirs(first_label_path,exist_ok=True)
    os.makedirs(image_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)
    os.makedirs(trajectories_path, exist_ok=True)

    image_files = [f for f in os.listdir(background_path) if f.endswith('.jpg')] #or jpg
    image_num = len(image_files)

    background_images = [cv2.imread(os.path.join(background_path, f"{i}.jpg"), cv2.IMREAD_GRAYSCALE) for i in range(image_num)] #or jpg
    target_images = [cv2.imread(os.path.join(target_path, f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(1, 94)]

    results = generate_degraded_dataset(background_images, target_images, num_targets=20)
pro.close()

