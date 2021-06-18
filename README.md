# Object Detection

The object of this project is to train a face detection model, then given a picture will return pixel coordinates of face occurrences. The dataset consists of small images of faces and non-faces such as background, a small part of faces. The given images have a resolution of 48x48 pixels. So, I first train a convolutional neural network for a binary classification model and use it to detect faces on the entire images.

<br>

# Dataset

Image patches of fixed size (3x48x48) of faces and non-faces.


![example](https://user-images.githubusercontent.com/37695060/122594426-47d0b080-d067-11eb-867a-9a3abca22072.png)


<br>

# Designed Architecture

I implement a binary fully convolutional model. Since this network is fully convolutional, I can feed it with bigger images and obtain face scores for different locations of image patches all at once. Spatial output corresponds to face probability at different locations of the input image.

>Implement a binary convolutional model and train for 10 epochs with batch size of 64. Morever, optimize Cross Entropy Loss and use Adam optimizer with learning rate 3e-4.

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1)

<br>

# Face detector
If we feed the model with image of size H * W, and we get the outptu of size outH * outW, the output at coordinates (0,0) corresponds to the top left corner patch of the image (0,0,48,48) [coordinates: (x1,y1,x2,y2); where (x1,y1)-coordinates of top left corner of the patch and (x2,y2)-coordinates of bottom right corner of the patch]. The output at botton right coordinates (outH-1, outW-1) corresponds to input image patch (W-1-48, H-1-48, W-1, H-1) [bottom right corner]. The other correspondences can be computed proportionally.

```
threshold 
```
<table>
  <tr>
    <td> <img src="sample.png"  alt="1" width = 360px height = 640px ></td>

    <td><img src="img2.png" alt="2" width = 360px height = 640px></td>
   </tr> 
   <tr>
      <td><img src="./Scshot/cab_arrived.png" alt="3" width = 360px height = 640px></td>

      <td><img src="./Scshot/trip_end.png" align="right" alt="4" width = 360px height = 640px>
  </td>
  </tr>
</table>



# Improvement

To improve my model, I tuned my achitecture, hyper-parameters and regularization as follows:

- Training parameters:
  
  - changed learning rate: 0.003
  - changed batch size: 128
  - scheduling with step size 5 and gamma 0.1

- Network architecture:

  - normalization layers

- Model regularization:

  - data augmentation only using `RandomHorizontalFlip(0.5)` with 0.5 probability of the image being flipped
  - dropout with a dropout percentage 0.2 only after the second and the fourth activation function

<br>

In conclusion, the validation loss and accuracy have been improved compared to the point 1 as follows:

|        |Loss|Accuracy|
|:-------|:---|:-------|
|Before|0.1270|95.24|
|After |0.0958|96.44|

Hence, the validation accuracy increases `1.2` from `95.24` to `96.44` by using the improved model.

<br>

# Conclusion

Using the imporved model, I found that the best threshold in this case is 0.993

<br>

# Reference
Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Senior Member, IEEE, and Yu Qiao, Senior Member, IEEE, "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
