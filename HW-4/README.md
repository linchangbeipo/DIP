# 3D-Gaussian Splatting

## Installation

首先需要安装natsort库：  
`pip install natsort`  

然后安装pytorch3d库：  
`conda install -c fvcore -c iopath -c conda-forge fvcore iopath`  
`conda install pytorch3d -c pytorch3d`  

## Datasets

稀疏点云获取：  
`python mvs_with_colmap.py --data_dir data/chair`  
`python debug_mvs_by_projecting_pts.py --data_dir data/chair`  

## Train

模型训练：  
`python train.py --colmap_dir data/**** --checkpoint_dir data/****/checkpoints`  

## Results

[<video width="300" height="200" controls>
    <source src="data/chair/debug_rendering_chair.mp4" type="video/mp4">
</video>]
