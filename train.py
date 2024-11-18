from transformers import MobileNetV1ImageProcessor
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import tqdm
import logging
logging.basicConfig(level=logging.INFO)

batch_size = 64
dataset = load_dataset("ILSVRC/imagenet-1k",cache_dir="C:\\Users\san20\PycharmProjects\ImageNetCNN", trust_remote_code=True)
image_processor = MobileNetV1ImageProcessor()

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

