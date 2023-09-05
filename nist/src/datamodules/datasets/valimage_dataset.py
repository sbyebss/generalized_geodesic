import os

from PIL import Image
from torch.utils import data
from torchvision import transforms


def get_paths(opt):
    image_dir = opt.val_image_dir
    image_list = opt.val_image_list
    # pylint:disable=consider-using-with, unspecified-encoding
    names = open(image_list).readlines()
    filenames = list(map(lambda x: x.strip("\n") + opt.val_image_postfix, names))
    image_paths = list(map(lambda x: os.path.join(image_dir, x), filenames))
    filenames = list(map(lambda x: x.strip("\n") + opt.val_mask_postfix, names))
    mask_paths = list(map(lambda x: os.path.join(opt.val_mask_dir, x), filenames))
    return image_paths, mask_paths


class ValImageDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        image_paths, mask_paths = get_paths(opt)

        self.image_paths = image_paths
        self.mask_paths = mask_paths

        size = len(self.image_paths)
        self.dataset_size = size
        transform_list = [
            transforms.Resize(
                (opt.crop_size, opt.crop_size), interpolation=Image.NEAREST
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.image_transform = transforms.Compose(transform_list)
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (opt.crop_size, opt.crop_size), interpolation=Image.NEAREST
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert("RGB")
        w, h = image.size
        image_tensor = self.image_transform(image)
        # mask image
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask = mask.convert("L")
        mask = mask.resize((w, h))
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0).float()
        input_dict = {
            "image": image_tensor,
            "mask": mask_tensor,
            "path": image_path,
        }

        return input_dict
