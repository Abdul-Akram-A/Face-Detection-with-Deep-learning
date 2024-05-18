# Real-Time Face Detection Using Deep Learning
Welcome to the Realtime Face Detection project! This repository contains code and resources for implementing a real-time face detection system using deep learning techniques.

### Project Description:
The Real-Time Face Detection project is an application of deep learning techniques, specifically utilizing a modified version of the VGG16 model, combined with custom output layers. The aim is to accurately detect faces in images or video streams in real-time.

### Model Architecture:
The model architecture is designed to process input images of size 120x120 pixels with RGB color channels. Here's a breakdown of the architecture:

1. **Input Layer:**
   - The model begins with an input layer that accepts images of size 120x120 pixels with three color channels (RGB).

2. **VGG16 Backbone:**
   - Utilizes the VGG16 pre-trained model, initialized with ImageNet weights, serving as the backbone of the network for feature extraction.

3. **Feature Extraction:**
   - Global Max Pooling layers are applied to extract significant features from the convolutional layers of the VGG16 network.

4. **Classification Head:**
   - Features extracted from the backbone are flattened and passed through a dense layer with 2048 ReLU activation units.
   - Responsible for classifying the presence of a face in the input image.
   - Outputs a single neuron with a sigmoid activation function, indicating the probability of a face being present.

5. **Regression Head:**
   - Another branch of the network extracts features using Global Max Pooling and flattens them.
   - Processed through a dense layer with 2048 ReLU activation units, followed by another dense layer with 4 neurons and a sigmoid activation function.
   - Predicts the bounding box coordinates (x, y, width, height) of the detected face.

6. **Model Output:**
   - The model produces two sets of predictions:
     - **Classification Output:** Probability of a face being present in the input image.
     - **Regression Output:** Bounding box coordinates of the detected face.

### Conclusion:
The Real-Time Face Detection project combines the strengths of deep learning and computer vision to create a powerful solution for accurate face detection in various environments. With the VGG16 backbone and custom output layers, the model efficiently processes images in real-time, making it suitable for applications such as video surveillance and facial recognition systems.
