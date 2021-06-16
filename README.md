# Object Detection

The object of this project is to train a face detection model, then given a picture will return pixel coordinates of face occurrences. The dataset consists of small images of faces and non-faces such as background, a small part of faces. The given images have a resolution of 48x48 pixels. So, I first trained a convolutional neural network for a binary classification model and use it to detect faces on the entire images.

# Dataset

Image patches of fixed size (3x48x48) of faces and non-faces.

# Designed Architecture

I implemented a binary fully convolutional model. Since this network is fully convolutional, I can feed it with bigger images and obtain face scores for different locations of image patches all at once. Spatial output corresponds to face probability at different locations of the input image.

# Training

# Evaluation

# Conclusion
