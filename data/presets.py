import omnivore.data.transforms as CT  # custom transforms
import torch
import torchvision.transforms as T
from omnivore.data.rand_aug3d import RandAugment3d
from torchvision.transforms.functional import InterpolationMode


# Video presets
class VideoClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        transform_list = [
            T.ConvertImageDtype(torch.float32),
            T.Resize(resize_size),
        ]
        if hflip_prob > 0:
            transform_list.append(T.RandomHorizontalFlip(hflip_prob))
        transform_list.extend(
            [
                T.Normalize(mean=mean, std=std),
                T.RandomCrop(crop_size),
                CT.ConvertTCHWtoCTHW(),
            ]
        )
        self.transforms = T.Compose(transform_list)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
    ):
        self.transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Resize(resize_size),
                T.Normalize(mean=mean, std=std),
                T.CenterCrop(crop_size),
                CT.ConvertTCHWtoCTHW(),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)
