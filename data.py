from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder


def transform():
    return Compose([
        ToTensor(),  ## 操作将图像转换为PyTorch张量，并将像素值归一化到0到1之间。
    ])


def get_training_set(data_dir, label_dir, patch_size, data_augmentation):
    return DatasetFromFolder(data_dir, label_dir, patch_size, data_augmentation, transform=transform())


def get_eval_set(data_dir, label_dir):
    return DatasetFromFolderEval(data_dir, label_dir, transform=transform())
