import torch

# TODO: Extremely hacky. Change ASAP.
import sys
sys.path.insert(0, './src/utils')
import utils


# TODO: Generally, make function names full and then shorten the names of
# variabels they are assigned to.
# TODO: Loss should not carry information about images.
class YOLOLoss:
    def __init__(self, image_width=448, image_height=448, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        self.image_width = image_width
        self.image_height = image_height
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = S
        self.B = B
        self.C = C
    
    def __call__(self, prediction, target):
        target = self.unique_detections_per_cell(target)
        # TODO: Make this less patchy.
        if len(target.shape) == 1:
            torch.unsqueeze(target, 0)

        coordinate_loss = self.coord_loss(prediction, target)
        object_noobject_loss = self.obj_noobj_loss(prediction, target)
        classes_loss = self.class_loss(prediction, target)

        return coordinate_loss + object_noobject_loss + classes_loss
    
    def coord_loss(self, prediction, target):
        sum_loss = 0.0
        for obj in target:
            curr_cell = obj[0]
            i = torch.div(curr_cell, self.S, rounding_mode='floor').long()
            j = (curr_cell - (i * self.S)).long()

            x_true, y_true, w_true, h_true = obj[1:5]

            _, max_b = utils.find_max_iou(i, j, prediction, x_true, y_true, h_true, w_true, self.image_width, self.image_height, self.S, self.B)

            # taking parameters for the box with max IoU

            x_box, y_box, w_box, h_box = prediction[i, j, max_b * 5:max_b * 5 + 4]


            # print('print', x_box, y_box, w_box, h_box, x_true, y_true, w_true, h_true)
            # TODO: check if max_iou should be in this formula.
            # TODO: Try without hardcoding forcing positive width and height.
            sum_loss += (x_box - x_true) ** 2 + (y_box - y_true) ** 2 + (torch.sqrt(w_box) - torch.sqrt(w_true)) ** 2 + (
                    torch.sqrt(h_box) - torch.sqrt(h_true)) ** 2 + 1000 * (abs(w_box) - w_box) + 1000 * (abs(h_box) - h_box)

        return self.lambda_coord * sum_loss


    def obj_noobj_loss(self, prediction, target):
        curr_cell = -1
        no_more_targets = len(target) == 0
        if not no_more_targets:
            curr_cell = target[0][0]
        curr_index = 0

        obj_sum_loss = 0
        noobj_sum_loss = 0

        for i in range(self.S):
            for j in range(self.S):
                cell_num = i * self.S + j
                ######## obj loss ########
                if cell_num == curr_cell and not no_more_targets:
                    x_true, y_true, w_true, h_true = target[curr_index][1:5]

                    # calc max IoU
                    _, max_b = utils.find_max_iou(i, j, prediction, x_true, y_true, w_true, h_true, self.image_width, self.image_height, self.S, self.B)

                    # calc obj loss for box with max IoU
                    confidence_box = prediction[i, j, max_b * 5 + 4]
                    obj_sum_loss += (1 - confidence_box) ** 2

                    curr_index += 1
                    if (curr_index == len(target)):
                        no_more_targets = True
                    else:
                        curr_cell = target[curr_index][0]
                ######## noobj loss ########
                else:
                    # TODO check for noobj loss -> too strict?
                    for b in range(self.B):
                        confidence_box = prediction[i, j, b * 5 + 4]
                        noobj_sum_loss += (0 - confidence_box) ** 2

        return obj_sum_loss + self.lambda_noobj * noobj_sum_loss


    def class_loss(self, prediction, target):
        sum_loss = 0
        for obj in target:
            curr_cell = obj[0]
            i = torch.div(curr_cell, self.S, rounding_mode='floor').long()
            j = (curr_cell - (i * self.S)).long()

            # these are one hot vectors
            ground_truth_classes = obj[5:]
            predicted_classes = prediction[i, j, 5 * self.B:]

            sum_loss += torch.sum((ground_truth_classes - predicted_classes) ** 2)

        return sum_loss
    
    def unique_detections_per_cell(self, target):
        # TODO: Check if this can happen at all. If it can, take maximal IoU.
        unique = target[0]
        curr = target[0][0]
        flag = 0
        for item in target:
            if item[0] != curr:
                flag = 1
                unique = torch.stack((unique, item), dim=0)
                curr = item[0]

        if flag: return unique
        return torch.unsqueeze(unique, dim=0)
