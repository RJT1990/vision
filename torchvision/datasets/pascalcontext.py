import os
import sys
import tarfile
import collections
from .vision import VisionDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from .utils import download_url, check_integrity

ARCHIVE_DICT = {
    'trainval_annot': {
        'url': 'https://codalabuser.blob.core.windows.net/public/trainval_merged.json',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
    },
    'trainval': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2010',
    }
}


class PASCALContext(VisionDataset):
    """`Pascal Context <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 split='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(PASCALContext, self).__init__(root, transforms, transform, target_transform)

        base_dir = ARCHIVE_DICT['trainval']['base_dir']
        self.voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        self.split = split

        if download:
            self.download()

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    @property
    def annotations_file(self):
        return os.path.join(self.voc_root, 'trainval_merged.json')

    def download(self):

        if not os.path.isdir(self.voc_root):
            archive_dict = ARCHIVE_DICT['trainval']
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=self.root,
                                         md5=archive_dict['md5'])
        else:
            msg = ("You set download=True, but a folder VOCdevkit already exist in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")
            print(msg)

        if not check_integrity(self.annotations_file):
            archive_dict = ARCHIVE_DICT['trainval_annot']
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=os.path.join(self.root, 'VOCdevkit', 'VOC2010'),
                                         md5=archive_dict['md5'])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
