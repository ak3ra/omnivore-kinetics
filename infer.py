import os
import json
from PIL import Image
import numpy as np

import torch
from torchvision.io import read_video, _read_video_from_file
import torchvision.transforms as T

import sys
# We add the folder: "<torchmultimodal_repo>/examples" to path to import the presets
sys.path.append(os.path.dirname(os.path.abspath("")))

import torchmultimodal.models.omnivore as omnivore
from omnivore.data import presets

from IPython.display import Video
from matplotlib import pyplot as plt



# Get model with pretrained weight
model = omnivore.omnivore_swin_t(pretrained=True)
model = model.eval()

# Read class list and video
with open("assets/kinetics400_class.json", "r") as fin:
    kinetics400_classes = json.load(fin)
video, audio, info = read_video("assets/kinetics400_val_snowboarding_001.mp4", output_format="TCHW")

# Since we sampled at 16 fps for training, and the input video is 30 fps
# we resample every 2 frames so it become 15 fps and closer to training fps
video = video[::2]

# Use first 50 frames
video = video[:50]

# Apply transforms
video_val_presets = presets.VideoClassificationPresetEval(crop_size=224, resize_size=224)
input_video = video_val_presets(video)
# Add batch dimension
input_video = input_video.unsqueeze(0)


# Get top5 labels
preds = model(input_video, "video")
top5_values, top5_indices = preds[0].topk(5)
top5_labels = [kinetics400_classes[index] for index in top5_indices.tolist()]
top5_labels
