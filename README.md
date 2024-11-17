# Traffic Detection Model BY Rohan Chatse(Data Scientists)

## Overview

This project implements a deep learning-based traffic detection model using Convolutional Neural Networks (CNN) with TensorFlow/Keras. The model is trained to classify images into two categories:

- **0** - No traffic (Clear road)
- **1** - Traffic present

This model uses a convolutional neural network to identify and classify traffic in images, and it is trained on a dataset of road images. The model can be used for real-time traffic monitoring or other applications requiring image-based traffic detection.

## Requirements

The following libraries are required to run the code:

- **TensorFlow** 2.x
- **NumPy**
- **OpenCV**
- **Matplotlib** (optional for visualization)

You can install the dependencies via pip:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

## Dataset

The dataset is expected to be organized in two directories:

- `data/train_ds`: Contains the training images.
- `data/test_ds`: Contains the validation (testing) images.

The images should be organized into subdirectories within the `train_ds` and `test_ds` folders, where each subdirectory corresponds to a different class (e.g., "traffic" and "no_traffic").

- **Traffic images** are labeled as class 1.
- **Non-traffic images** are labeled as class 0.

Example directory structure:

```
data/
  train_ds/
    traffic/
      image1.jpg
      image2.jpg
      ...
    no_traffic/
      image1.jpg
      image2.jpg
      ...
  test_ds/
    traffic/
      test_image1.jpg
      test_image2.jpg
      ...
    no_traffic/
      test_image1.jpg
      test_image2.jpg
      ...
```

## Model Architecture

The traffic detection model is built using a Convolutional Neural Network (CNN) with the following architecture:

1. **Conv2D Layer (32 filters)**: 3x3 kernel, ReLU activation, padding='valid'
2. **MaxPooling2D**: 2x2 pool size
3. **Conv2D Layer (64 filters)**: 3x3 kernel, ReLU activation, padding='valid'
4. **MaxPooling2D**: 2x2 pool size
5. **Conv2D Layer (128 filters)**: 3x3 kernel, ReLU activation, padding='valid'
6. **MaxPooling2D**: 2x2 pool size
7. **Flatten Layer**: Flatten the output for fully connected layers
8. **Dense Layer (128 units)**: ReLU activation
9. **Dense Layer (64 units)**: ReLU activation
10. **Dense Layer (1 unit)**: Sigmoid activation (output between 0 and 1)

The model is compiled using the Adam optimizer with binary crossentropy loss function for binary classification. The metric used is accuracy.

## Training the Model

To train the model, use the following code:

```python
history = model.fit(train_ds, epochs=15, validation_data=validation_ds)
```

This will train the model for 15 epochs. You can adjust the number of epochs and other parameters according to your needs.

## Saving the Model

Once the model is trained, it is saved to a file named `traffic_detector_main.h5` for later use:

```python
model.save('traffic_detector_main.h5')
```

## Testing the Model

You can test the trained model on new images by loading the model and predicting the class of a given image. Here is an example of how to use the model to make predictions on a single image:

```python
import cv2

# Load the trained model
model = tf.keras.models.load_model('traffic_detector_main.h5')

# Load and preprocess the image
testimg = cv2.imread('testimg1.jpeg')
testimg = cv2.resize(testimg, (256, 256))  # Resize to match the input size
testinput = testimg.reshape((1, 256, 256, 3))  # Reshape for the model

# Make the prediction
prediction = model.predict(testinput)

# Interpret the result
if prediction >= 0.5:
    print("Traffic detected (Class 1)")
else:
    print("No traffic detected (Class 0)")
```

You can replace `'testimg1.jpeg'` with the path to any image you want to test.

## Example Images

The following images are used in the script to test the model:

- `traffictest.jpeg`: Image with traffic
- `clearroads.jpeg`: Image with no traffic
- `cr2.jpeg`: Image with traffic
- `notraffic1.jpeg`: Image with no traffic
- `testimg1.jpeg`: Another test image

These images are resized to 256x256 pixels before passing through the model for prediction.

## Notes

- Ensure that the images used for training and testing are preprocessed correctly. In this implementation, images are normalized by dividing by 255 to scale pixel values between 0 and 1.
- The model currently works with binary classification, specifically for detecting traffic vs. no traffic.
- You can adjust the model architecture or training parameters based on your specific requirements.

## Conclusion

This model provides a simple yet effective way to classify images of roads as either having traffic or being clear. It can be further refined by using more data, experimenting with model architectures, or applying data augmentation techniques.
