from roboflow import Roboflow
import requests
from PIL import Image
from io import BytesIO  
import matplotlib.pyplot as plt
import matplotlib.patches as patches

rf = Roboflow(api_key="xWcdwnSFEpUBswBeChuZ")
project = rf.workspace().project("dark-pattern-_visual-detection")
model = project.version(1).model
img_path="https://assets.econsultancy.com/images/0008/4654/zsl_focused.jpg"
prediction_data = model.predict(img_path, hosted=True, confidence=20, overlap=30)

all_predictions = []

for prediction in prediction_data:

    x = prediction['x']
    y = prediction['y']
    width = prediction['width']
    height = prediction['height']
    confidence = prediction['confidence']
    class_name = prediction['class']
    image_path = prediction['image_path']
    single_prediction = {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'confidence': confidence,
        'class': class_name,
        'image_path': image_path
    }
    all_predictions.append(single_prediction)
image_url = all_predictions[0]['image_path']
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
fig, ax = plt.subplots()
ax.imshow(image)
for prediction in all_predictions:
    x = prediction['x']
    y = prediction['y']
    width = prediction['width']
    height = prediction['height']
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.savefig('image_with_boxes.png', bbox_inches='tight', pad_inches=0.1)

plt.show()
