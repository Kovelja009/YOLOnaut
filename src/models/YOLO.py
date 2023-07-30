# TODO: Import only exactly what is needed.
import torch
import torchvision
import math
import cv2

classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor'
]

class ResNet50Bottom(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x

class YOLO(torch.nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        # TODO: Pass argument to decide on pretraining.
        self.backbone = ResNet50Bottom(original_model=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT))
        self.pool = torch.nn.Conv2d(2048, 1024, (1, 1))
        self.conv = torch.nn.Sequential(
            # TODO: Parameterize numbers of channels.
            torch.nn.Conv2d(1024, 1024, (3, 3)),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(1024, 1024, (3, 3), stride=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(1024, 1024, (3, 3)),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(1024, 1024, (3, 3)),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 4096),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 7 * 7 * 30),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.conv(x)
        # TODO: Reshape to (S, S, 5 * B + C)
        return x