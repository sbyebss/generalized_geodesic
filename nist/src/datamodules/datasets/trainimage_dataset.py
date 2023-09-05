import os

from PIL import Image
from torch.utils import data

from src.datamodules.datasets.transforms import get_params, get_transform

# FIXME: customize path


def get_paths(opt):
    image_dir = opt.train_image_dir
    image_list = opt.train_image_list
    # pylint:disable=consider-using-with, unspecified-encoding
    names = open(image_list).readlines()
    filenames = list(map(lambda x: x.strip("\n") + opt.train_image_postfix, names))
    image_paths = list(map(lambda x: os.path.join(image_dir, x), filenames))
    return image_paths


class TrainImageDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        image_paths = get_paths(opt)

        self.image_paths = image_paths

        size = len(self.image_paths)
        self.dataset_size = size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        try:
            image_path = self.image_paths[index]
            image = Image.open(image_path)
            image = image.convert("RGB")
            params = get_params(self.opt, image.size)
            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image)
            input_dict = {
                "image": image_tensor,
                "path": image_path,
            }
            return input_dict
        except:  # pylint: disable=bare-except
            print(f"skip {image_path}")
            return self.__getitem__((index + 1) % self.__len__())
