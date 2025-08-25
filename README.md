# Pseudo-Sample Generation and Hierarchical Fusion-Association Tracking for Dense Thermal Infrared UAVs
<img width="8996" height="1554" alt="跟踪整体框架" src="https://github.com/user-attachments/assets/ee7a5ffd-ee14-4fcf-b944-5a0c44ebdb0a" />

## Pseudo-Sample Generation Module

This folder contains the code for generating pseudo-samples for thermal infrared UAV tracking.

### 1. Data Preparation
| Dataset Type | Download Link | Extraction Code |
|--------------|---------------|-----------------|
| Visible-light target samples | [Baidu Netdisk](https://pan.baidu.com/s/1hqhfbyttdnw7pXdCiIOIgg) | `enso` |
| IRT-B background samples | [Baidu Netdisk](https://pan.baidu.com/s/1ak4Cth-aBuAkDtK3X0oq8A) | `enso` |
| IRC-B background samples | [Baidu Netdisk](https://pan.baidu.com/s/1pKLkE1cHMM5-FJH6TxyyPA) | `enso` |

### 2. Infrared Sample Generation
Run the following script to convert visible-light targets to infrared-style samples: 
bash 
python Target_Sample_Collection.py
> Note: For batch processing, modify the script parameters according to your needs.

### 3. Pseudo-Sample Generation
#### 3.1 Preparation
- Place converted infrared targets in `target_images_ir/` folder
#### 3.2 Generation Command
bash
python Pseudo-Sample_Generation.py
output/
├── images/ # Generated pseudo-images
└── labels/ # YOLO format annotations

### 4. Dependencies
- FastReID: [Official GitHub](https://github.com/JDAI-CV/fast-reid)
- BoT-SORT: [Official GitHub](https://github.com/NirAharon/BOT-SORT)
- YOLOv12: [Official GitHub](https://github.com/sunsmarterjie/yolov12)
- UAVSwarm dataset: [Official GitHub](https://github.com/UAVSwarm/UAVSwarm-dataset)
<img width="2500" height="753" alt="Strong-R-mulit(1)" src="https://github.com/user-attachments/assets/13f1b059-11ca-4064-a0c6-ad43437cf9ac" />

### 5. Citation
If you use this code in your research, please cite our paper.
