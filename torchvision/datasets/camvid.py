import json
import os
from collections import namedtuple
import zipfile

from .utils import extract_archive
from .vision import VisionDataset
from PIL import Image


class CamVid(VisionDataset):
    """`CamVid <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/>`_ Dataset, where the format of the dataset
    is the same as in this repo by Cambridge MLG: https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

    Args:
        root (string): Root directory of dataset where directory ``val/valannot`` or ``train/trainannot`` or
        ``test/testannot`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get dataset for training

        .. code-block:: python

            dataset = CamVid('./data/camvid', split='train')

            img, target = dataset[0]

        Get dataset for validation

        .. code-block:: python

            dataset = CamVid('./data/camvid', split='val')

            img, target = dataset[0]
    """

    CamVidClass = namedtuple('CamVidClass', ['name', 'color'])

    classes = [
        CamVidClass('sky', (128, 128, 128)),
        CamVidClass('building', (128, 0, 0)),
        CamVidClass('pole', (192, 192, 128)),
        CamVidClass('road_marking', (255, 69, 0)),
        CamVidClass('road', (128, 64, 128)),
        CamVidClass('pavement', (60, 40, 222)),
        CamVidClass('tree', (128, 128, 0)),
        CamVidClass('sign_symbol', (192, 128, 128)),
        CamVidClass('fence', (64, 64, 128)),
        CamVidClass('car', (64, 0, 128)),
        CamVidClass('pedestrian', (64, 64, 0)),
        CamVidClass('bicyclist',  (0, 128, 192)),
        CamVidClass('unlabeled', (0, 0, 0)),
    ]

    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        super(CamVid, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(self.root, split)
        self.targets_dir = os.path.join(self.root, split + 'annot')
        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            dataset_zip = os.path.join(self.root, 'CamVid.zip')  # if CamVid.zip is present, then auto unzip

            if os.path.isfile(dataset_zip):
                extract_archive(from_path=dataset_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" are inside the "root" directory')

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, file_name))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index][i])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)