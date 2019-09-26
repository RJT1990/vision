import PIL
from sotabencheval.semantic_segmentation import PASCALVOCEvaluator
import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as transforms
import tqdm

from sotabench_transforms import Normalize, Compose, Resize, ToTensor

MODEL_NAME = 'fcn_resnet101'

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

device = torch.device('cuda')

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
my_transforms = Compose([Resize((520, 480)), ToTensor(), normalize])

dataset_test = torchvision.datasets.VOCSegmentation(root='./.data/vision/voc2012', year='2012', image_set="val", 
                                                    transforms=my_transforms, download=True)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=32,
    sampler=test_sampler, num_workers=4,
    collate_fn=collate_fn)

model = torchvision.models.segmentation.__dict__['fcn_resnet101'](num_classes=21, pretrained=True)
model.to(device)
model.eval()

evaluator = PASCALVOCEvaluator(model_name='FCN (ResNet-101)',
                              paper_arxiv_id='1605.06211')

with torch.no_grad():
    for image, target in tqdm.tqdm(data_loader_test):
        image, target = image.to('cuda'), target.to('cuda')
        output = model(image)
        output = output['out']
        
        evaluator.add(output.argmax(1).flatten().cpu().numpy(), target.flatten().cpu().numpy())
        if evaluator.cache_exists:
            break
        
evaluator.save()
