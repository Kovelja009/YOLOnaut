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

# def collate_batch(batch):
   
#     ret = torch.zeros_like((len(batch), len(batch[0])))
#     for (_tensor,_targets) in batch:
#         label_list.append(_label)
#         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
#         text_list.append(processed_text)


#     return text_list.to(device),label_list.to(device)

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

voc_dataset = torchvision.datasets.VOCDetection(local_dataset_path,
                                             image_set='train',
                                             transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((448, 448)), # TODO: Don't hardcode this.
                                                torchvision.transforms.ToTensor(),
                                             ]),
                                             download=False,
                                            #  target_transform=preprocess.ParseDict(S=7),
                                             )
# print(voc_dataset.__getitem__(1))
# exit()
# Train, test, validation split
train_val_split = 0.8
cnt_train = int(len(voc_dataset) * train_val_split)
cnt_val = len(voc_dataset) - cnt_train
train_dataset, val_dataset = torch.utils.data.random_split(voc_dataset, [cnt_train, cnt_val])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=True, collate_fn=lambda x: x)

# print('train_len', len(train_dataloader), 'val_len', len(val_dataloader))

device = 'cuda'

yolo = YOLO.YOLO()
yolo.to(device)

optimizer = torch.optim.SGD(yolo.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6#, verbose=True
                                                       )

import time
# with torch.no_grad():
#     inputs = torch.tensor(np.expand_dims(np.array(voc_dataset.__getitem__(0)[0]), 0))

# inputs = inputs.to(device)
# targets = torch.tensor(voc_dataset.__getitem__(0)[1])
# targets = targets[targets[:, 0].sort()[1]]
# targets.to(device)

torch.set_printoptions(threshold=sys.maxsize)


# print(len(yolo.state_dict()))
# torch.save(yolo.state_dict(), '/home/dels/Documents/YOLOnaut/models/initial_model.pth')


S = 7
B = 2
C = 20
model = YOLO.YOLO()
model.load_state_dict(torch.load('/home/dels/Documents/YOLOnaut/models/initial_model.pth'))
print(voc_dataset.__getitem__(1)[1])
outputs = model(torch.unsqueeze(voc_dataset.__getitem__(1)[0], 0))
def save_evaluations(image_name, predictions):
    image_path = os.path.join('/home/dels/Documents/YOLOnaut/data/VOCdevkit/VOC2012/JPEGImages', image_name)
    image_with_evaluation = utils.draw_image(image_path, predictions)

    save_path = image_path.replace('JPEGImages', 'JPEGEvaluations')
    cv2.imwrite(save_path, image_with_evaluation)
print(outputs.reshape((S, S, 5 * B + C)))
save_evaluations('2008_000015.jpg', outputs.reshape((S, S, 5 * B + C)))
exit()
from contextlib import redirect_stdout
# orig_stdout = sys.stdout
# f = open('/home/dels/Desktop/out.txt', 'w')
# sys.stdout = f

best_val_loss = 1e6

loss_class = yolo_loss.YOLOLoss()
with open('/home/dels/Desktop/out.txt', 'a') as f:
    with redirect_stdout(f):
        print('started training')
start = time.time()
for epoch in range(50):
    with open('/home/dels/Desktop/out.txt', 'a') as f:
        with redirect_stdout(f):
            print('epoch ', epoch, ' started')

    yolo.train()
    for i, batch in enumerate(train_dataloader):
        # Forward pass
        for batch_element in batch:
            loss = 0
            with torch.no_grad():
                inputs = torch.unsqueeze(batch_element[0], 0).clone().detach()
                inputs = inputs.to(device)
                # print('inputs shape', inputs.shape)
                targets = torch.tensor(batch_element[1])
                targets = targets.to(device)
                # print('targets shape', targets.shape)

            outputs = yolo(inputs)
            loss += loss_class(outputs.reshape((loss_class.S, loss_class.S, 5 * loss_class.B + loss_class.C)), targets)

            ## IMPORTANT PRINT
            # print(loss)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()
            # scheduler.step(epoch_loss)
        with open('/home/dels/Desktop/out.txt', 'a') as f:
            with redirect_stdout(f):
                print('trained batch ', i , ' with average loss: ', loss.item() / len(batch))


    yolo.eval()
    with torch.no_grad():
        val_loss = 0
    for i, val_element in enumerate(val_dataloader):
        with torch.no_grad():
            # print('sejp', val_element)
            val_inputs = torch.unsqueeze(val_element[0][0], 0).clone().detach()
            val_inputs = val_inputs.to(device)
            # print('inputs shape', inputs.shape)
            val_targets = torch.tensor(val_element[0][1])
            val_targets = val_targets.to(device)
            # print('targets shape', targets.shape)

            val_outputs = yolo(val_inputs)
            val_loss += loss_class(val_outputs.reshape((loss_class.S, loss_class.S, 5 * loss_class.B + loss_class.C)), val_targets)

    with open('/home/dels/Desktop/out.txt', 'a') as f:
        with redirect_stdout(f):
            print('average validation loss: ', val_loss.item() / len(val_dataloader))
    if(val_loss < best_val_loss):
        with open('/home/dels/Desktop/out.txt', 'a') as f:
            with redirect_stdout(f):
                print('saving model after epoch ', epoch)
        torch.save(yolo.state_dict(), '/home/dels/Documents/YOLOnaut/models/initial_model.pth')
        best_val_loss = val_loss

    with open('/home/dels/Desktop/out.txt', 'a') as f:
        with redirect_stdout(f):
            print('epoch ', epoch, ' finished')

end = time.time()
with open('/home/dels/Desktop/out.txt', 'a') as f:
    with redirect_stdout(f):
        print('finished training')
        print(end - start)
# S = 7
# B = 2
# C = 20
# print(outputs.reshape((S, S, 5 * B + C)))
# sys.stdout = orig_stdout
# f.close()