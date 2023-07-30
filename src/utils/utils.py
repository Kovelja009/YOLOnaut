import cv2
import numpy as np
import torch

# TODO: Do not hardcode this here.
classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor'
]

# TODO: Too many arguments, pass three: coordinates, dimensions and cell.set
def convert_to_absolute(x, y, w, h, i, j, image_width, image_height, S):
    cell_width = image_width / S
    cell_height = image_height / S

    w = w * image_width
    h = h * image_height
    x = j * cell_width + x * cell_width
    y = i * cell_height + y * cell_height
    return x, y, w, h


# TODO: Too many arguments, pass three: coordinates, dimensions and cell.set
def calc_iou(prediction_coord, target_coord, i, j, image_width, image_height, S):
    x1, y1, w1, h1 = prediction_coord
    x2, y2, w2, h2 = target_coord

    x1, y1, w1, h1 = convert_to_absolute(x1, y1, w1, h1, i, j, image_width, image_height, S)
    x2, y2, w2, h2 = convert_to_absolute(x2, y2, w2, h2, i, j, image_width, image_height, S)

    # upper left points
    x1_upper, y1_upper = x1 - w1 / 2, y1 - h1 / 2
    x2_upper, y2_upper = x2 - w2 / 2, y2 - h2 / 2

    # lower right points
    x1_lower, y1_lower = x1 + w1 / 2, y1 + h1 / 2
    x2_lower, y2_lower = x2 + w2 / 2, y2 + h2 / 2

    # intersection points
    x_upper_inter = max(x1_upper, x2_upper)
    y_upper_inter = max(y1_upper, y2_upper)

    x_lower_inter = min(x1_lower, x2_lower)
    y_lower_inter = min(y1_lower, y2_lower)

    if (x_upper_inter > x_lower_inter) or (y_upper_inter > y_lower_inter):
        return 0

    intersection_area = (x_lower_inter - x_upper_inter) * (y_lower_inter - y_upper_inter)
    union_area = w1 * h1 + w2 * h2 - intersection_area

    return intersection_area / union_area

# TODO: Too many arguments required in general, this should be abstractized.
# Probably make IoUUtils class which will initialize all these values and then
# call its methods.

def find_max_iou(i, j, prediction, x_true, y_true, w_true, h_true, img_width, img_height, S, B):
    max_iou = 0
    max_b = 0
    for b in range(B):
        x_box, y_box, w_box, h_box = prediction[i, j, b * 5:b * 5 + 4]
        iou = calc_iou((x_box, y_box, w_box, h_box), (x_true, y_true, w_true, h_true), i, j, img_width, img_height, S)
        if iou >= max_iou:
            max_iou = iou
            max_b = b
    return max_iou, max_b

# TODO: The following functions should be separated into class VisualUtils.

def show_image(image, predictions, image_width, image_height, S):
    image = show_grid(image, S, image_width, image_height)
    image = show_objects(image, predictions, image_width, image_height)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_grid(image, grid_size):
    # print('X',image.shape, grid_size, image_width, image_height, 'X')
    # Step 1: Calculate grid size and spacing
    rows, cols, _ = image.shape
    grid_size = 7
    grid_spacing_rows = rows // (grid_size)
    grid_spacing_cols = cols // (grid_size)
    image = np.ascontiguousarray(image)
    # Step 2: Draw grid lines on the image
    for i in range(1, grid_size):
        # Draw horizontal lines
        y = i * grid_spacing_rows
        cv2.line(image, (0, y), (cols, y), (0, 255, 0), 1)

        # Draw vertical lines
        x = i * grid_spacing_cols
        cv2.line(image, (x, 0), (x, rows), (0, 255, 0), 1)
    
    return image

# Visualization utils.

# lambda_coord = 5
# lambda_noobj = 0.5

# # object_treshold = 0.1

# S = 7
# B = 2
# C = 20
# img_width = 448
# img_height = 448


# (x,y,w,h,c) for each bounding box

# every cell has 2 bounding boxes
# prediction = .rand((S, S, B * 5 + C))

def convert_to_apsolute(x, y, w, h, i, j, img_width=448, img_height=448, S=7):
    cell_width = img_width / S
    cell_height = img_height / S

    w = w * img_width
    h = h * img_height
    x = j * cell_width + x * cell_width
    y = i * cell_height + y * cell_height
    return x, y, w, h


def load_image(path, img_width=448, img_height=448):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_width, img_height))
    return img


def draw_image(path, predictions=None, S=7):
    img = load_image(path)
    img = show_grid(img, S)
    img = show_objects(img, predictions)
    return img
    # cv2.imshow("YOLOnaut", np.ascontiguousarray(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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


def show_objects(img, predictions, object_threshold=0.6, S=7, B=2):
    for i in range(S):
        for j in range(S):
            highest_class_probability = max(predictions[i, j, B * 5:])
            if highest_class_probability >= object_threshold:
                box_confidences = []
                for b in range(B):
                    box_confidences.append(predictions[i, j, b * 5 + 4])
                max_box_confidence = max(box_confidences)
                max_box_index = box_confidences.index(max_box_confidence)
                x_relative, y_relative, w_relative, h_relative = predictions[i, j,
                                                                 max_box_index * 5:max_box_index * 5 + 4]
                if w_relative <= 0 or h_relative <= 0:
                    continue
                x, y, w, h = convert_to_apsolute(x_relative, y_relative, w_relative, h_relative, i, j)

                # upper left point
                x_upper, y_upper = x - w / 2, y - h / 2

                # lower right point
                x_lower, y_lower = x + w / 2, y + h / 2

                classes_hot_vector = predictions[i, j, B * 5:]
                class_index = classes_hot_vector.argmax()
                class_name = classes[class_index.item()]
                print('----------------------------------------')
                print(f'{class_name}, highest_class_prob:{round(highest_class_probability.item(), 2)}')
                print(f'box: i:{i}, j:{j}, b:{max_box_index}')
                print(f'absolute -> width:{round(w.item(), 2)}, height:{round(h.item(), 2)}')
                print(f'relative -> width:{round(w_relative.item(), 2)}, height:{round(h_relative.item(), 2)}')
                print(f'relative -> x:{round(x_relative.item(), 2)}, y:{round(y_relative.item(), 2)}')
                print(f'absolute -> x:{round(x.item(), 2)}, y:{round(y.item(), 2)}')
                print('----------------------------------------')
                cv2.rectangle(img, (int(x_upper), int(y_upper)), (int(x_lower), int(y_lower)), (0, 0, 255), 2)
                cv2.putText(img, f'{class_name}, conf:{round(highest_class_probability.item(), 2)}',
                            (int(x_upper), int(y_upper - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

    return img

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


def draw_ground_truth(img, target, S=7):
    img = show_grid(img, S)
    img = show_gt(img, target)
    # print(img)
    cv2.imshow("YOLOnaut", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# classes = {
#     0: 'aeroplane',
#     1: 'bicycle',
#     2: 'bird',
#     3: 'boat',
#     4: 'bottle',
#     5: 'bus',
#     6: 'car',
#     7: 'cat',
#     8: 'chair',
#     9: 'cow',
#     10: 'dining table',
#     11: 'dog',
#     12: 'horse',
#     13: 'motorbike',
#     14: 'person',
#     15: 'potted plant',
#     16: 'sheep',
#     17: 'sofa',
#     18: 'train',
#     19: 'tv/monitor'
# }

# img = load_image("/home/dels/Documents/YOLOnaut/src/utils/third.jpg")
# # print(img)
# # print(img.shape)
# # print(prediction.shape)
# draw_image(img, predictionnnnnn)
