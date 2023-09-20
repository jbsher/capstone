# HaGRID Hand Gesture Recognition Dataset
![Alt Text](https://github.com/hukenovs/hagrid/blob/master/images/hagrid.jpg?raw=true)

## Problem Statement

With the rise of digital communication platforms and smart home systems, there's an increasing need for efficient hand gesture recognition (HGR) systems that can be integrated across different applications. Our objective is to take a subset of the expansive HaGRID dataset to devise a HGR system that can recognize and categorize various hand gestures in diverse environments.

To achieve our goal, we will utilize the annotations provided in the form of bounding boxes, landmarks, and associated metadata. By the end of our project, our aim is to have an HGR system that is not just accurate but is also versatile, making it seamlessly integrable across platforms like video conferencing tools, home automation interfaces, and automotive systems.

## Dataset

**HaGRID - Hand Gesture Recognition Image Dataset**

HaGRID is a vast repository of FullHD RGB images encompassing 18 distinct hand gestures, collected from a diverse group of individuals spanning various ages and backgrounds. These images encapsulate a plethora of scenarios ranging from different indoor lighting conditions to more challenging setups like having subjects positioned against a window. Additionally, the dataset provides the complexity of recognizing gestures even when there's another non-gesturing hand present in the frame, as represented by the "no_gesture" class.

**Images:**
- Subset of 1,000 images per category, totaling 18,000 images (from 552,992 total FullHD images available).
- Dataset includes 34,730 unique persons and scenes with variations in age, lighting, and conditions.
- Subjects aged 18 to 65, indoors, and up to 4 meters from the camera.

**Annotations:**
- Bounding boxes of hands with gesture labels in COCO format [top left X position, top left Y position, width, height].
- 21 landmarks in format [x, y] relative image coordinates.
- Leading hand markings (left or right for gesture hand) and leading hand confidence (leading_conf).
- User_id field for dataset splitting.

## Preprocessing

- Reduced dataset to 1,000 images per category for train/validation.
- Resized images to 512 x 512.
- Normalized values between [-1, 1].
- Combined image and annotation data into a JSON file.
- Created a TensorFlow tfrecord file.
- Created a PyTorch dataset.

## Exploratory Data Analysis (EDA)

- Displayed images for inspection in each category with RGB values.
- Inspected images with corresponding bounding boxes.
- Analyzed the distribution of bounding box widths and heights.
- Created a heatmap of bounding box start coordinates.
- Analyzed the distribution of the leading hand in the picture.
- Created a heatmap of mean distances between landmarks.

## Modeling

-Dataset Preparation: The dataset consists of hand gesture images with annotations for different gesture classes, including 'call,' 'fist,' 'peace,' and more.

-Data Augmentation: Image data augmentation techniques were applied, such as resizing, normalization, and converting to tensors, to prepare the dataset for training.

-Custom Dataset Class: A custom GestureDataset class was implemented to handle data loading, annotation parsing, and transformation.

-Model Selection: The chosen model architecture is the SSDLite320 with MobileNetV3 backbone, pretrained on a large-scale dataset. It was adapted for gesture recognition with class-specific output heads.

-Transfer Learning: Transfer learning was applied by initializing the model with pretrained weights, which boosts training efficiency.

-Training and Validation Split: The dataset was split into training and validation sets based on user IDs to ensure diverse representation in both sets.

-Training Loop: The model was trained using stochastic gradient descent (SGD) with a learning rate scheduler and warmup phase. Losses were computed for object detection tasks.

-Evaluation Metrics: Evaluation metrics, including mean average precision (MAP) and loss were computed on a separate test dataset to assess model performance.

Model Saving: Model checkpoints were saved at the end of each epoch to track training progress and enable model reusability. The model ran for 10 epochs.

Results and Metrics: The modeling process yielded insights into gesture recognition performance, as indicated by MAP, precision, recall, and F1-score metrics, which are critical for assessing the model's practical use.

## User Interface (UI)

- Created a Streamlit app that allows a user to upload an image in .jpg, .jpeg, .png formats and get a prediction with a boundary box.

## Conclusion

- RGB values in descending order are red, green, blue.

- The distribution of bounding box heights was slightly higher than bounding box width, as to be expected with hand gestures.

- The heatmap of bounding box starting locations had two major hotspots, one on the upper left and one on the upper right, with two smaller hotspots below them. Also this is to be expected if photos are generally taken with people standing in the middle of the frame.

- The left hand is the leading hand in the picture roughly 1000 more times than right leading hand.

- Mean pairwise landmark distances were closer towards the center as compared to the edges.

- The test loss during training converged around epoch 6 at around 0.08.

- After running our training, we were able to create a model with a MAP of 0.37. We then created a simple Streamlit app that showcases the functionality of the model.

## Reflection/Improvements

- This basic model could be optimized if the entire 522,992 image dataset was included. Other factors include cost and time constraints posed by renting GPU, as well as inexperience on my part.

- Other pretrained models should be tested on the data for further testing.

- Recommend taking a very small sample of pictures to create a model to cut down on time and cost, then including the entire dataset once everything is running as expected.

