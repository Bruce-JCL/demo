from random import random
import cv2
import numpy as np

coco_class = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

random_color_list = np.random.randint(0, 255, (10, 3), dtype=np.uint8)

def draw_ssd_result(img, all_boxes):
    '''
    检测结果绘制
    :param img: 需绘制的img
    :param all_boxes: [N, 5], ymin, xmin, ymax, xmax, cls
    :param cost_time: 耗时ms
    :return 绘制完成的img
    '''
    h, w, _ = img.shape
    img = img.astype(np.uint8)
    color_step = int(255/len(all_boxes))
    for i in range(len(all_boxes)):
        ymin, xmin, ymax, xmax, cls = all_boxes[i]
        if ymin == ymax:
            continue
        left, right, top, bottom = (xmin * w, xmax * w,
                                  ymin * h, ymax * h)
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        cv2.putText(img, coco_class[min(int(cls), len(coco_class)-1)], (left, int(top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (left, top), (right, bottom), (int(random_color_list[i][0]), int(random_color_list[i][1]), int(random_color_list[i][2])), thickness = 1)
    return img
    
def preprocess_image_for_tflite_uint8(image, model_image_size=300):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model_image_size, model_image_size))
    image = np.expand_dims(image, axis=0)
    return image

def non_max_suppression(scores, boxes, classes, max_boxes=10, min_score_thresh=0.5):
    '''
    非极大值抑制算法
    :param scores: 每个检测框的分数
    :param boxes: 每个检测框的坐标
    :param classes: 每个检测狂的类别id
    :param max_boxes: 最多检测数量
    :param min_score_thresh: 分数阈值
    return tuple(分数, 坐标, 类)
    '''
    out_boxes = []
    out_scores = []
    out_classes = []
    if not max_boxes:
        max_boxes = boxes.shape[0]
    for i in range(min(max_boxes, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            out_boxes.append(boxes[i])
        else:
            out_boxes.append(np.zeros_like(boxes[i]))
        out_scores.append(scores[i])
        out_classes.append(classes[i])

    out_boxes = np.array(out_boxes)
    out_scores = np.array(out_scores)
    out_classes = np.array(out_classes)

    return out_scores, out_boxes, out_classes