from sklearn.preprocessing import LabelBinarizer

## TODO: Utils Class? Preprocessing.

classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor'
]

def class_to_onehot(_class, classes):
    """TODO: Decription."""
    lb = LabelBinarizer()

    lb.fit(classes)

    return lb.transform([_class])[0]

def bndbox_to_yolo(bndbox, image_width, image_height, S):
    """TODO: Decription."""
    xmax, xmin, ymax, ymin = int(bndbox['xmax']), int(bndbox['xmin']), int(bndbox['ymax']), int(bndbox['ymin'])
    
    cell_width = image_width / S
    cell_height = image_height / S

    x = (xmax + xmin) / 2
    cell_x = int(x / cell_width)
    x_rel = (x - (cell_width) * (x // (cell_width))) / (cell_width)

    y = (ymax + ymin) / 2
    cell_y = int(y / cell_height)
    y_rel = (y - (cell_height) * (y // (cell_height))) / (cell_height)

    cell = cell_y * S + cell_x
    
    # TODO: Check if abs is needed.
    w_rel = abs(xmax - xmin) / image_width
    h_rel = abs(ymax - ymin) / image_height

    return cell, x_rel, y_rel, w_rel, h_rel

class ParseDict(object):
    """TODO: Class description."""
    def __init__(self, S):
        self.S = S

    def __call__(self, target):
        image_width = int(target['annotation']['size']['width'])
        image_height = int(target['annotation']['size']['height'])
        target = target['annotation']['object']
        for i, s in enumerate(target):
            bndbox = s['bndbox']
            _class = s['name']
            target[i] = (*bndbox_to_yolo(bndbox, image_width, image_height, self.S), *class_to_onehot(_class, classes))

        return target


class ParseDataset(object):
    def __init__(self, image_size, S):
        self.image_size = image_size
        self.S = S
