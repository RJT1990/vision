from torchvision.models.alexnet import alexnet
import torchvision.transforms as transforms
from torchbench.image_classification import ImageNet
import PIL
import torch

# Define Transforms    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
b0_input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=alexnet(pretrained=True),
    paper_model_name='AlexNet',
    input_transform=b0_input_transform,
    batch_size=256,
    num_gpu=1
)
