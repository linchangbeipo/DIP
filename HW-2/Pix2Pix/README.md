# Pix2Pix by FCN

## Installation
To install requirements:  `python -m pip install -r requirements.txt`

Then click [Pytorch](https://pytorch.org), install pytorch-cuda=12.1

## Datasets
To prepare the datasets(you can choose one dataset):  
`bash download_***_dataset.sh`

## Train
To train the model:  
`python train.py`

## Results
在训练了250个轮次后，损失值达到了0.125，预测图像如下：

<figure class = "half">
<img src="./assets/result_1.png" width="900">
</figure>

<figure class = "half">
<img src="./assets/result_2.png" width="900">
</figure>
