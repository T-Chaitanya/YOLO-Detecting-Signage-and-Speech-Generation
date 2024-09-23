import cv2
import os

from PIL import Image
from ultralytics import YOLO
# from PIL
import cv2
import numpy as np

def check_and_convert_image(image):
    # Get the number of channels
    channels = image.shape[2] if len(image.shape) == 3 else 1

    # If image has 4 channels (RGBA), convert it to 3 (RGB)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # If image is grayscale (no channels), convert it to 3 channel
    elif channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    # Define COCO Labels
    if labels == []:
        labels = {
            0: 'accessibility', 1: 'door', 2: 'elevator sign', 3: 'elevator', 4: 'exit sign',
            5: 'fire alarm', 6: 'fire extinguisher', 7: 'handle', 8: 'left arrow', 9: 'men-s washroom',
            10: 'person', 11: 'push handle', 12: 'right arrow', 13: 'stair sign', 14: 'trash can',
            15: 'water dispenser', 16: 'women-s washroom'
        }

    # Define colors
    if colors == []:
        colors = [
            (89, 161, 197), (67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11),
            (190, 76, 98), (130, 172, 179), (115, 209, 128), (204, 79, 135), (136, 126, 185),
            (209, 213, 45), (44, 52, 10), (101, 158, 121), (179, 124, 12), (25, 33, 189),
            (45, 115, 11), (73, 197, 184), (62, 225, 221)
        ]

    # Plot each box
    for box in boxes:
        # Add score in label if score=True
        if score:
            label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%"
            print(label)
        else:
            label = labels[int(box[-1])]
        # Filter every box under conf threshold if conf threshold set
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

    # Show image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    image = np.copy(image)

    # Draw bounding box
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    # Add label text
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

path = "indoor-yolo-v8/test/images/"
test_images = os.listdir("indoor-yolo-v8/test/images/")

model = YOLO('/Users/chaitanyatata/PycharmProjects/Yolo_v8_Model_FP/runs/detect/train3/weights/best.pt')

for i in range(0, 2):
    # image = Image.open(path + test_images[random.randint(0, 200)])
    image = Image.open(path + test_images[i])
    # image = Image.open("/content/drive/My Drive/444.png")
    image = np.asarray(image)
    image = check_and_convert_image(image)
    results = model.predict(image, imgsz=640)
    labels = ['accessibility', 'door', 'elevator sign', 'elevator', 'exit sign', 'fire alarm', 'fire extinguisher',
              'handle', 'left arrow', 'men-s washroom', 'person', 'push handle', 'right arrow', 'stair sign',
              'trash can', 'water dispenser', 'women-s washroom']
    # colors = [[255, 0, 255], [123, 0, 255], [255, 0, 12], [123, 255, 0]]
    ann_image = plot_bboxes(image, results[0].boxes.data, labels, colors=[], score=True, conf=0.70)
    ann_image = cv2.cvtColor(ann_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', ann_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
