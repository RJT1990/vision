from .lsun import LSUN, LSUNClass
from .folder import ImageFolder, DatasetFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .stl10 import STL10
from .mnist import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST
from .svhn import SVHN
from .phototour import PhotoTour
from .fakedata import FakeData
from .semeion import SEMEION
from .omniglot import Omniglot
from .sbu import SBU
from .flickr import Flickr8k, Flickr30k
from .voc import VOCSegmentation, VOCDetection
from .ade20k import ADE20K
from .camvid import CamVid
from .cityscapes import Cityscapes
from .pascalcontext import PASCALContext
from .imagenet import ImageNet
from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .sbd import SBDataset
from .vision import VisionDataset
from .usps import USPS

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder', 'DatasetFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
           'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
           'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
           'VOCSegmentation', 'VOCDetection', 'ADE20K', 'CamVid', 'Cityscapes', 'ImageNet', 'PASCALContext'
           'Caltech101', 'Caltech256', 'CelebA', 'SBDataset', 'VisionDataset',
           'USPS')
