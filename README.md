# Overview
In this project, I aim to implement U-Net and V-Net (a 2D version actually) to perform semantic segmentation on the M2NIST dataset.

[M2NIST](https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist) is a multi-digit version of the MNIST dataset which has segmentation masks for each image. It is a relatively simple dataset for semantic segmentation tasks, making it easy and fast to train on.

In the following sections, I will briefly explain the main ideas and present the results of each model in the assigned task.

# Semantic Segmentation
Semantic segmentation is a computer vision task that involves classifying each pixel in an image into a specific category or class. This task goes beyond object detection, which only identifies the presence of objects in an image without specifying their exact boundaries. In semantic segmentation, the goal is to label each pixel with a class label (e.g., person, car, road, sky) to provide a detailed understanding of the image's content. This technique is widely used in various fields such as autonomous driving, medical imaging, and satellite imagery analysis. By segmenting an image semantically, computers can better understand the visual world and make more informed decisions based on the pixel-wise annotations.

| <img src="data:image/jpeg;base64,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" width="400" height="100"> | 
|:--:| 
| *Semantic Segmentation* |
*source : [Nanonets](https://www.google.com/url?sa=i&url=https%3A%2F%2Fnanonets.com%2Fblog%2Fsemantic-image-segmentation-2020%2F&psig=AOvVaw2nq_S2KglaqVXqMC6QOHWL&ust=1708576854268000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCOiUgvbOu4QDFQAAAAAdAAAAABAE)*

# Fully Convolutional Network
Convolutional Neural Networks (CNNs) traditionally excel in tasks like image classification by leveraging convolutional and pooling layers to extract features and make predictions. When it comes to semantic segmentation, Fully Convolutional Networks (FCNs) adapt the CNN architecture by removing the fully connected layers. FCNs maintain spatial information throughout the network, allowing input images of varying sizes to produce output segmentation maps of the same dimensions.

In the encoding stage, FCNs extract high-level features through convolutional and pooling layers, gradually reducing spatial dimensions. The decoding stage involves upsampling the spatial dimensions back to the original image size using transposed convolutions or upsampling layers. Through decoding, FCNs combine low-level features from encoding with high-level features, often through skip connections, to enhance segmentation accuracy. This process enables FCNs to perform pixel-wise predictions for semantic segmentation effectively.

| <img src="https://d1m75rqqgidzqn.cloudfront.net/wp-data/2020/10/19185819/image-43.png" width="300" height="100"> | 
|:--:| 
| *FCN 8* |
*source : [GreatLearning](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mygreatlearning.com%2Fblog%2Ffcn-fully-convolutional-network-semantic-segmentation%2F&psig=AOvVaw0H42f57eMcbHEWUbggWhoF&ust=1708579247928000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCNDesO7Xu4QDFQAAAAAdAAAAABAE)*

# U-Net
U-Net is an architecture designed for biomedical image segmentation tasks but widely used in various image segmentation applications due to its effectiveness. U-Net improves upon the basic FCN architecture by introducing skip connections that directly connect encoding layers to corresponding decoding layers at the same spatial resolution. These skip connections aid in preserving fine-grained details during upsampling and help avoid information loss in the decoding process.

By incorporating skip connections that directly link feature maps from encoding stages to corresponding decoding stages, U-Net encourages the efficient fusion of high-level semantic information with detailed spatial information. This strategy enables the network to access features at multiple levels of abstraction during the upsampling process, aiding in the reconstruction of intricate spatial structures and enhancing segmentation accuracy.

The symmetrical design of the U-Net architecture ensures that the lengths of the downsampling and upsampling paths match, promoting a balance between feature extraction and resolution refinement. This symmetry facilitates the seamless integration of skip connections, allowing extracted features from different scales to be effectively combined and propagated across the network for more robust segmentation performance.

| <img src="images/u.png" width="400" height="300"> | 
|:--:| 
| *U-Net* |
*source : [Medium](https://www.google.com/url?sa=i&url=https%3A%2F%2Flukmanaj.medium.com%2Fu-net-architecture-revolutionizing-computer-vision-through-innovative-image-segmentation-e1e2155c38b1&psig=AOvVaw3_sP1q-PTTwuApImsbYL7f&ust=1708579770169000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCODJ2uXZu4QDFQAAAAAdAAAAABAQ)*

# V-Net
This architecture closely follows the principles of U-Net by incorporating two symmetric contracting and expanding paths. However, V-Net distinguishes itself by adopting a fully-convolutional structure, exclusively employing convolution operations while eliminating pooling layers.

The decision to forgo pooling layers in V-Net serves two primary purposes. Firstly, pooling operations can be replaced efficiently with convolutions of larger strides, resulting in networks that are quicker to train. Additionally, utilizing deconvolutions for upsampling in the expanding path is more straightforward compared to unpooled layers, facilitating easier comprehension and analysis.

One notable departure between U-Net and V-Net is the choice of training methodology. While U-Net typically utilizes stochastic gradient descent, V-Net leverages residual connections to expedite convergence and enhance segmentation outcomes.

By incorporating residual connections, V-Net enhances the flow of gradients during training, mitigating issues related to vanishing gradients and enabling faster convergence. This approach allows the model to learn more effectively and efficiently, improving segmentation accuracy.

The elimination of pooling layers in V-Net reduces memory overhead and helps maintain spatial information throughout the network. This absence of pooling layers also enhances the model's ability to extract robust features and preserves finer details in the segmentation process.

| <img src="images/v.png" width="400" height="300"> | 
|:--:| 
| *V-Net* |
*source : [Towards Data Science](https://www.google.com/url?sa=i&url=https%3A%2F%2Ftowardsdatascience.com%2Freview-v-net-volumetric-convolution-biomedical-image-segmentation-aa15dbaea974&psig=AOvVaw3B_lGotPgHdbwdb9tYMOya&ust=1708580996516000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCKj1mK3eu4QDFQAAAAAdAAAAABAE)*

# Results
First of all, i want to recall that i've implemented the V-Net for 2D images and actually i used the bold ideas in V-Net to modify U-Net, so there may be room for improving the architecture.

First of all, I want to recall that I've implemented the V-Net for 2D images, and actually, I used the bold ideas in V-Net to modify U-Net, so there may be room for improving the architecture.

Before comparing the results, I want to briefly explain what I have done. In the downsampling block, I have replaced the pooling layer with a simple conv2d layer with a stride of 2. In order to add residual activation to the output of the convolution layers, we need to increase the number of channels, so I used a 1x1 convolution layer. Its activation function is linear, and I believe it is the best choice for the following reasons:

Linear activation (or no activation) in the 1x1 convolution does not introduce nonlinearities to the transformation. This property allows the layer to act as a simple feature mapping tool without altering the input distribution significantly. As a result, the layer primarily focuses on adjusting the channel dimensionality while retaining information from the previous layer. This approach aids in preserving the identity of the input while learning additional representations in parallel. Therefore, the model will learn to bypass the input of 1x1 conv after enough iterations. In the upsampling block, I used transposed convolutions to upsample the previous layer output. A residual connection is used in this block too. The rest of the implementation is the same.

Both models performed well on the dataset. Adding more iterations appeared to reduce the loss significantly. To ensure a fair comparison, I aimed to reach a similar validation loss for both models. The V-Net showed more accuracy in spatial segmentation. For instance, in a test image, the V-Net excelled in outlining the boundaries of a digit 1.

| <img src="images/vnet_test0.jpg" width="400" height="400"> | <img src="images/test0.jpg" width="400" height="400"> |
|:--:|:--:|
| *V-Net output* | *U-Net output* |

| class | V-Net IoU | U-Net IoU | V-Net Dice score | U-Net Dice score|
| ----- | ---- | ---- | ----- | ---- |
| 0 | 0.9457 | 0.9253 | 0.9670 | 0.9684 |
| 1 | 0.9478 | 0.9259 | 0.9646 | 0.9774 |
| 2 | 0.9413 | 0.9475 | 0.9670 | 0.9700 |
| 3 | 0.9208 | 0.9338 | 0.9504 | 0.9658 |
| 4 | 0.9122 | 0.9282 | 0.9389 | 0.9653 |
| 5 | 0.9421 | 0.9345 | 0.9686 | 0.9617 |
| 6 | 0.9511 | 0.9444 | 0.9738 | 0.9762 |
| 7 | 0.9506 | 0.9082 | 0.9719 | 0.9460 |
| 8 | 0.9348 | 0.9419 | 0.9562 | 0.9733 |
| 9 | 0.9175 | 0.8846 | 0.9461 | 0.9390 |
| 10 | 0.9975 | 0.9973 | 0.9987 | 0.9987 |

The results indicate that V-Net better captures the spatial relationships and boundaries within these classes.
