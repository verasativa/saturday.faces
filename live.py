import cv2, time, os
from align import detector
from PIL import Image
from head import metrics
from backbone import model_irse
import torch
import numpy as np
from matplotlib import cm

# Colors
colors = cm.get_cmap('tab10').colors
colors = (
    colors[4],
    colors[6],
    colors[9],
    colors[8],
    colors[1],
    # El resto
    colors[0],
    colors[2],
    colors[3],
    colors[5],
    colors[7],
)
colors = np.array(colors) * 255
# Debug
import pickle, os

img_side = 112

# Models
#

# Set device
if torch.cuda.is_available():
    print('GPU available; working on GPU')
    DEVICE = torch.device("cuda:0")
else:
    print('GPU not available; working on CPU')
    DEVICE = torch.device("cpu")

# Get fist models, whatever filename
backbone_filename, head_filename = False, False
for filename in os.listdir('model'):
    if not backbone_filename:
        if filename[:8] == 'Backbone':
            backbone_filename = 'model/' + filename
    elif not head_filename:
        if filename[:4] == 'Head':
            head_filename = 'model/' + filename
    else:
        break

# Backbone
backbone = model_irse.IR_SE_50([img_side, img_side])
backbone.load_state_dict(torch.load(backbone_filename, map_location='cpu'))
backbone.eval()

# Head
head = metrics.ArcFace(in_features = 512, out_features = 7, device_id = None)
head.load_state_dict(torch.load(head_filename, map_location='cpu'))
head.eval()

# Labels
labels = [
    'Alicia',
    'Alma',
    'Josefa',
    'Maca',
    'Marisol',
    'Silvana',
    'Vera'
]

# Face resize
def prepare_face(face, img_side = 112):
    shape = np.array(face.shape[:2])
    maxdim = shape.argmax(axis=0)
    rate = img_side / shape[maxdim]
    new_shape = (shape * rate).astype('int')
    delta = np.array([img_side, img_side]) - new_shape
    delta_a = delta // 2
    delta_b = delta - delta_a
    resized = cv2.resize(face, dsize=tuple(new_shape.tolist()), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(resized, delta_a[1], delta_b[1], delta_a[0], delta_b[0], cv2.BORDER_CONSTANT)


# Video capture
#
if os.environ['USER'] == 'N':
    device = 1
else:
    device = 0
capture = cv2.VideoCapture(device)

frameRate_start_time = time.time()
frameRate_refresh = 1 # displays the frame rate every 1 second
frameRate_counter = 0
frameRate_fps = 0

# Window
window = 'main_win'
cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


while(True):
    #Capture frame
    ret, img = capture.read()

    # Get frameRate
    frameRate_counter+=1
    if (time.time() - frameRate_start_time) > frameRate_refresh :
        frameRate_fps = frameRate_counter / (time.time() - frameRate_start_time)
        frameRate_counter = 0
        frameRate_start_time = time.time()

    # Faces location
    bounding_boxes, landmarks = detector.detect_faces(Image.fromarray(img))
    for i, box in enumerate(bounding_boxes):
        # Set color
        color = colors.astype('int')[i % 10].tolist()
        # Copy just the face
        box = box.astype('int')
        face = img[box[1]:box[3], box[0]:box[2]]

        # Build the face frame
        corner_top_left     = (int(box[0]), int(box[1]))
        corner_buttom_right = (int(box[2]), int(box[3]))
        cv2.rectangle(img, corner_top_left, corner_buttom_right, color, 2)

        with torch.no_grad():
            # Fake batch
            face = prepare_face(face)
            if face.shape[0] == img_side and face.shape[1] == img_side:
                face = np.expand_dims(face, axis=0)
                #print(face.shape)
                face = torch.from_numpy(np.transpose(face, (0, 3, 1, 2)))
                face = face.float()
                embedings = backbone(face)
                result = head(embedings, label=None)

                # print(result)
                probs = torch.nn.functional.softmax(result, dim=1)
                index = int((torch.abs((torch.max(probs).item() - probs)) < 0.0001).nonzero()[0, 1])
                human_probs = (torch.max(probs).item() - (1 / 7)) * 6/7 * 100
                text = '{} {:.2f}%'.format(labels[index], human_probs)
                # print('Prediccion: {}'.format(labels[index]))
                x_pos = box[0] + (box[2] - box[0]) // 3
                y_pos = box[1] - (box[3] - box[1]) // 16
                pos = (x_pos, y_pos)
                cv2.putText(img, text, pos,
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=1,
                            color=color)
            else:
                x_pos = box[0] + (box[2] - box[0]) // 3
                y_pos = box[1] - (box[3] - box[1]) // 16
                pos = (x_pos, y_pos)
                cv2.putText(img, '{}'.format(face.shape), pos,
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=1,
                            color=color)
    # print(bounding_boxes)

    # Write frameRate
    pos = (int(img.shape[1] - 165), int(img.shape[0] - 10))
    cv2.putText(img, "{0:.2f} fps".format(frameRate_fps), pos,
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1,
                color=(10, 10, 10))

    #Show image
    cv2.imshow(window, img)
    #cv2.imshow('img', img)

    #Quit with q keyPress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #Full screen with f
    if cv2.waitKey(1) & 0xFF == ord('f'):
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
capture.release()
cv2.destroyAllWindows()
