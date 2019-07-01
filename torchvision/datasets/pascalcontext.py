import json
import numpy as np
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
from .utils import check_integrity, download_and_extract_archive, download_url

ARCHIVE_DICT = {
    'trainval_annot': {
        'url': 'https://codalabuser.blob.core.windows.net/public/trainval_merged.json',
        'md5': '3c2d0c0656b7be9eb61928ffe885d8ce',
    },
    'trainval': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
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
        image_dir = os.path.join(self.voc_root, 'JPEGImages')
        mask_dir = os.path.join(self.voc_root, 'SegmentationClass')
        self.split = split

        if download:
            self.download()

        if not os.path.isdir(self.voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')


        self.annotations_dict = json.load(open(dataset.annotations_file, 'r'))
        self.ids = annotations_dict['images']

        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        self._key = np.array(range(len(self._mapping))).astype('uint8')
        mask_file = os.path.join(self.voc_root, self.split+'.pth')
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            print('Mask file does not exist...now preprocessing masks')
            self.masks = self._preprocess_masks(mask_file)

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess_masks(self, mask_file):
        masks = {}
        print(len(self.ids))
        for i in range(len(self.ids)):
            print(i)
            img_id = self.ids[i]
            #mask = Image.fromarray(self._class_to_index(self._get_mask(img_id)))
            #masks[img_id['image_id']] = mask
        #torch.save(masks, mask_file)
        return masks

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
            download_url(archive_dict['url'], self.root,
                         filename=os.path.basename(archive_dict['url']),
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
