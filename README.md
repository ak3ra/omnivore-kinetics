A Single Model for Many Visual Modalities


Experimental project to train and evaluate the Omnivore multimodal model. The model is designed to work with images, videos, and even RGB-D data, sharing the same encoder and using modality-specific heads for classification. 

Features
* Multimodal Input: Supports images, videos, and single-view 3D (RGB-D) data.
* Flexible Training: Uses gradient accumulation for video inputs to manage GPU memory.
* Mixed Precision: Optional mixed precision training using torch.cuda.amp.
* Distributed Training: Built-in support for multi-GPU training.
* EMA: Optional exponential moving average for smoother training.

