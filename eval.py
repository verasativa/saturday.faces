from head import metrics
from backbone import model_irse
import torch, os
from PIL import Image
import numpy as np

# Set device
if torch.cuda.is_available():
    print('GPU available; working on GPU')
    DEVICE = torch.device("cuda:0")
else:
    print('GPU not available; working on CPU')
    DEVICE = torch.device("cpu")

#print(type(DEVICE))
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

head = metrics.ArcFace(in_features = 512, out_features = 7, device_id = None)
head.load_state_dict(torch.load(head_filename, map_location='cpu'))
head.eval()


backbone = model_irse.IR_SE_50([112, 112])
backbone.load_state_dict(torch.load(backbone_filename, map_location='cpu'))
backbone.eval()

labels = [
    'Alicia',
    'Alma',
    'Josefa',
    'Maca',
    'Marisol',
    'Silvana',
    'Vera'
]


# Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))

    # Get the dimensions of the new image size
    width, height = img.size

    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 112) / 2
    top = (height - 112) / 2
    right = (width + 112) / 2
    bottom = (height + 112) / 2
    img = img.crop((left, top, right, bottom))

    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img / 255

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485) / 0.229
    img[1] = (img[1] - 0.456) / 0.224
    img[2] = (img[2] - 0.406) / 0.225

    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis, :]

    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image



import os

excluded_files = ['.DS_Store']
base_dir = '/Users/N/repositories/datasets/saturday.faces/faces.align/'
for person in os.listdir(base_dir):
    if person not in excluded_files:
        print('\nGround truth: {}'.format(person))
        for image in os.listdir(base_dir + person):
            if image not in excluded_files:
                print('File name: {}'.format(image))
                image_path = base_dir + person + '/' + image

                face = process_image(image_path)

                embedings = backbone(face)
                result = head(embedings, label=None)
                #print(result)
                probs = torch.nn.functional.softmax(result, dim=1)
                index = int((torch.abs((torch.max(probs).item() - probs)) < 0.0001).nonzero()[0,1])
                print('Prediccion: {}'.format(labels[index]))
                #probs = probs - probs.min()
                #print(probs)

#for label in range(0, len(labels)):
 #   results = head(embedings, label=torch.tensor([label]))
  #  print(labels[label])
   # print(results)