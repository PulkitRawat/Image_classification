from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy
from pathlib import Path

path = Path("chest_xray")
path.__dir__()

data = ImageDataLoaders.from_folder(path,train = "train", valid = "test", ds_tfms=aug_transforms(do_flip=False), size=224, bs=64, num_workers=8)