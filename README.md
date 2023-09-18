# HaGRID Hand Gesture Recognition Dataset
![Alt Text](https://github.com/hukenovs/hagrid/blob/master/images/hagrid.jpg?raw=true)

## Problem Statement

With the rise of digital communication platforms and smart home systems, there's an increasing need for efficient hand gesture recognition (HGR) systems that can be integrated across different applications. Our objective is to harness the potential of the expansive HaGRID dataset to devise a robust HGR system that can recognize and categorize various hand gestures in diverse environments.

HaGRID is a vast repository of FullHD RGB images encompassing 18 distinct hand gestures, collected from a diverse group of individuals spanning various ages and backgrounds. These images encapsulate a plethora of scenarios ranging from different indoor lighting conditions to more challenging setups like having subjects positioned against a window. Additionally, the dataset provides the complexity of recognizing gestures even when there's another non-gesturing hand present in the frame, as represented by the "no_gesture" class.

To achieve our goal, we will utilize the annotations provided in the form of bounding boxes, landmarks, and associated metadata. By the end of our project, our aim is to have an HGR system that is not just accurate but is also versatile, making it seamlessly integrable across platforms like video conferencing tools, home automation interfaces, and automotive systems.

## Dataset

**HaGRID - Hand Gesture Recognition Image Dataset**

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

- Utilized transfer learning with TensorFlow.
- Used the pre-trained ssdlite320 model as the starting point for the TensorFlow model.

## User Interface (UI)

**[Add details about your UI here]**

## Conclusion

**[Summarize your project's findings and outcomes]**

[Additional sections can be added as needed for your project's documentation.]

This README provides an overview of the HaGRID Hand Gesture Recognition Dataset project, including the problem statement, dataset details, preprocessing, exploratory data analysis, modeling, and user interface. For more detailed information and updates, refer to the project documentation and code.
