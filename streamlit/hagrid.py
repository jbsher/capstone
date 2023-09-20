import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
import io
import torchvision.models.detection as models
import os
import gdown
import pickle
import torchvision
from PIL import ImageFont



# Define the path to the model weights file in Google Drive
google_drive_url = "https://drive.google.com/file/d/1-Si7DetALvp65a3TAyJek9uBNPRMErG-/view?usp=sharing"

# Function to download the model weights from Google Drive
def download_model_weights():
    gdown.download(google_drive_url, "mdl_epoch_11.pth", quiet=False)

# Check if the model weights file exists locally, and if not, download it from Google Drive
if not os.path.exists("mdl_epoch_11.pth"):
    st.info("Downloading model weights from Google Drive. Please wait...")
    download_model_weights()
    
# Define the class names for gestures
class_names = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace_inverted',
   'peace',
   'rock',
   'stop_inverted',
   'stop',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture'
]
    
# Load trained model
model = models.ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=len(class_names) + 1)
model.load_state_dict(torch.load('mdl_epoch_11.pth', map_location=torch.device('cpu')))

st.title("Gesture Recognition App")

# Upload image
uploaded_image = st.file_uploader("Upload an image of a gesture", type=["jpg", "jpeg", "png"])

def decode_prediction(prediction, class_names, confidence_threshold=0.5):
    pred_boxes = []
    pred_labels = []
    pred_scores = []

    for p in prediction:
        # Filter out low-confidence predictions
        mask = p['scores'] > confidence_threshold

        pred_boxes += p['boxes'][mask].tolist()
        pred_labels += p['labels'][mask].tolist()
        pred_scores += p['scores'][mask].tolist()

    # Convert class indices to class labels
    pred_labels = [class_names[label] for label in pred_labels]

    return pred_boxes, pred_labels, pred_scores


from PIL import ImageDraw, ImageFont

def draw_bounding_boxes(image, boxes, labels):
    # Convert the image to RGB
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    # Use a default font (change the path to a TTF font file if needed)
    font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 15), label, font=font, fill="red")

    return image


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Recognize Gesture"):
        # 1. Preprocess the Image
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((320, 320)),  
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)  

        # Put the model in evaluation mode
        model.eval()

        # 2. Model Prediction
        with torch.no_grad():
            prediction = model(input_tensor)

        # 3. Post-process the Prediction
        pred_boxes, pred_labels, pred_scores = decode_prediction(prediction, class_names)  

        # 4. Display the Results
        image_with_boxes = draw_bounding_boxes(image, pred_boxes, pred_labels) 
        st.image(image_with_boxes, caption="Detected Gestures", use_column_width=True)
