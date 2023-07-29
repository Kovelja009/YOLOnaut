import torchvision
import torch
import numpy as np
import cv2

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
# print(voc_data.__getitem__(3)[1])
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

# VANJA DRAW GT
def convert_to_apsolute(x, y, w, h, i, j, img_width, img_height, S=7):
    cell_width = img_width / S
    cell_height = img_height / S

    w = w * img_width
    h = h * img_height
    x = j * cell_width + x * cell_width
    y = i * cell_height + y * cell_height
    return x, y, w, h


def show_gt(img, target, S=7):
    for obj in target: 
        cell_num = obj[0]
        i = torch.div(cell_num, S, rounding_mode='floor').long()
        j = (cell_num - (i * S)).long()
        
        x_rel, y_rel, w_rel, h_rel = obj[1:5]

        x, y, w, h = convert_to_apsolute(x_rel, y_rel, w_rel, h_rel, i, j, 448, 448)

        # upper left point
        x_upper, y_upper = x - w / 2, y - h / 2

        # lower right point
        x_lower, y_lower = x + w / 2, y + h / 2

        cv2.rectangle(img, (int(x_upper), int(y_upper)), (int(x_lower), int(y_lower)), (0, 0, 255), 2)
    
    return img

def show_grid(img, grid_size):
    # Step 1: Calculate grid size and spacing
    rows, cols, _ = img.shape
    grid_size = 7
    grid_spacing_rows = rows // (grid_size)
    grid_spacing_cols = cols // (grid_size)

    # Step 2: Draw grid lines on the image
    for i in range(1, grid_size):
        # Draw horizontal lines
        y = i * grid_spacing_rows
        cv2.line(img, (0, y), (cols, y), (0, 255, 0), 1)

        # Draw vertical lines
        x = i * grid_spacing_cols
        cv2.line(img, (x, 0), (x, rows), (0, 255, 0), 1)

    return img


def draw_groud_truth(img, target, S=7):
    img = show_grid(img, S)
    img = show_gt(img, target)
    # print(img)
    cv2.imshow("YOLOnaut", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# print(voc_data.__getitem__(3)[1])
# targets = voc_data.__getitem__(3)[1]
# img = utils.load_image('/home/dels/Documents/YOLOnaut/src/utils/third.jpg')
# # print('imggggg', img)
# draw_groud_truth(img, targets)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# exit()

yolo = YOLO.YOLO()
# optimizer = torch.optim.SGD(yolo.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.SGD(yolo.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6, verbose=True)

with torch.no_grad():
    inputs = np.expand_dims(np.array(voc_data.__getitem__(3)[0]), 0)

targets = torch.tensor(voc_data.__getitem__(3)[1])

torch.set_printoptions(threshold=sys.maxsize)

loss_class = yolo_loss.YOLOLoss()
for epoch in range(50):
    # Forward pass
    outputs = yolo(torch.tensor(inputs))
    # print('outputs', outputs.reshape((S, S, 5 * B + C)))
    # print('targets', targets)
    loss = loss_class(outputs.reshape((loss_class.S, loss_class.S, 5 * loss_class.B + loss_class.C)), targets)


    print(loss)
    # l.append(loss)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    # for param in yolo.parameters():
    #     print(param.grad)
    optimizer.step()
    # scheduler.step()
    # scheduler.step(epoch_loss)

S = 7
B = 2
C = 20
print(outputs.reshape((S, S, 5 * B + C)))
