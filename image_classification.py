from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy
from pathlib import Path
from PIL import Image
from fastai.vision.widgets import *

def is_cat(x):
    return x[0].isupper()

path = Path("chest_xray")
path.__dir__()

# data = ImageDataLoaders.from_folder(path,train = "train", valid = "test", ds_tfms=aug_transforms(do_flip=False), size=224, bs=64, num_workers=8)

# image = Image.open("chest_xray/train/normal/IM-0115-0001.jpeg")
# print(image.shape)
# image.show()
if __name__ == "__main__":
    tfms = aug_transforms(do_flip = True, flip_vert = False, mult=2.0)
    data= ImageDataLoaders.from_folder(path,train = "train", valid = "test", item_tfms=Resize(128), batch_tfms=tfms, bs = 30, num_workers = 4)

    learn = cnn_learner(data, resnet34, metrics = [accuracy])
    learn.fine_tune(4)

    learn.save("stage -1")
    learn.load("stage -1")

    # learn.unfreeze()
    # learn.lr_find()
    # learn.recorder.plot(suggestion = True)

    # learn.fit_one_cycle(2, max_lr = slice(3e-7,3e-6))

    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix(figsize = (12,12), dpi = 60)

    # cleaner = ImageClassifierCleaner(learn)
    # cleaner

    # learn.export()
    # learn_inf =  learn_inf = load_learner(path/'export.pkl')
    # learn_inf.predict("img")