import torchvision
import torch
import numpy as np
import cv2
import os

import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# TODO: Change ASAP.
local_dataset_path = '~/Documents/YOLOnaut/data'

import sys

sys.path.insert(0, './src/data')
import preprocess

sys.path.insert(0, './src/models')
import YOLO

sys.path.insert(0, './src/loss')
import yolo_loss

sys.path.insert(0, './src/utils')
import utils

voc_data = torchvision.datasets.VOCDetection(local_dataset_path,
                                             image_set='train',
                                             transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((448, 448)), # TODO: Don't hardcode this.
                                                torchvision.transforms.ToTensor(),
                                             ]),
                                             download=False,
                                             target_transform=preprocess.ParseDict(S=7),
                                             )
# print(voc_data.__getitem__(0)[1])
# exit()

# def draw_ground_truth(img, cell, x_rel, y_rel, w_rel, h_rel, S=7, image_width=448, image_height=448):
#     i = torch.div(cell, S, rounding_mode='floor').long()
#     j = (cell - (i * S)).long()
#     cell_width = image_width / S
#     cell_height = image_height / S
#     x = (j + x_rel) * cell_width
#     y = (i + y_rel) * cell_height

#     w = w_rel * image_width
#     h = h_rel * image_height

#     cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color=(0, 0, 255))

#     return img

# print(voc_data.__getitem__(0)[1])
# targets = voc_data.__getitem__(0)[1]
# img = utils.load_image('/home/dels/Documents/YOLOnaut/src/utils/horse.jpg')
# # print('imggggg', img)
# draw_groud_truth(img, targets)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# exit()

device = 'cuda'

yolo = YOLO.YOLO()
yolo.to(device)
# optimizer = torch.optim.SGD(yolo.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.SGD(yolo.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6, verbose=True)
import time
with torch.no_grad():
    inputs = torch.tensor(np.expand_dims(np.array(voc_data.__getitem__(0)[0]), 0))

inputs = inputs.to(device)
targets = torch.tensor(voc_data.__getitem__(0)[1])
targets = targets[targets[:, 0].sort()[1]]
targets.to(device)

torch.set_printoptions(threshold=sys.maxsize)

loss_class = yolo_loss.YOLOLoss()

# print('targets', targets.shape)
# print('inputs', inputs.shape)
start = time.time()
for epoch in range(50):
    # Forward pass
    outputs = yolo(inputs)
    # print('outputs', outputs.reshape((S, S, 5 * B + C)))
    # print('targets', targets)
    loss = loss_class(outputs.reshape((loss_class.S, loss_class.S, 5 * loss_class.B + loss_class.C)), targets)

    ## IMPORTANT PRINT
    # print(loss)
    # l.append(loss)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    # for param in yolo.parameters():
    #     print(param.grad)
    optimizer.step()
    # scheduler.step()
    # scheduler.step(epoch_loss)
end = time.time()
print(end - start)
S = 7
B = 2
C = 20
print(outputs.reshape((S, S, 5 * B + C)))

def save_evaluations(image_name, predictions):
    image_path = os.path.join('/home/dels/Documents/YOLOnaut/data/VOCdevkit/VOC2012/JPEGImages', image_name)
    image_with_evaluation = utils.draw_image(image_path, predictions)

    save_path = image_path.replace('JPEGImages', 'JPEGEvaluations')
    cv2.imwrite(save_path, image_with_evaluation)

save_evaluations('2008_000008.jpg', outputs.reshape((S, S, 5 * B + C)))