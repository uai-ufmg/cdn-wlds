# Implemented models

## **Bootstrap Your Own Latent (BYOL)**

**BYOL** (*Bootstrap Your Own Latent*) is a non-contrastive self-supervised learning method for representation learning without relying on negative pairs or contrastive learning techniques. It uses two distinct neural networks:

- **Online Network**: Learns representations by predicting the target network’s outputs of augmented views of the same image.
  
- **Target Network**: Provides stable outputs for the online network to learn from, updated gradually via an **exponential moving average** of the online network's parameters.

It does not specify architectures for the networks, so we used the Residual Network (ResNet).

#### Key Aspects

- **No Negative Pairs**: Unlike other self-supervised methods, BYOL does not require contrasting different images, eliminating the need for negative samples.
  
- **Two-Network System**: The interaction between the online and target networks allows for effective learning of representations without collapsing to trivial solutions.
  
- **Gradient Updates**: Only the online network is updated through backpropagation, while the target network updates smoothly, ensuring consistent training dynamics.


## **Dynamic Time Warping (DTW)**

**Dynamic Time Warping (DTW)** is a powerful algorithm used to measure the similarity between two temporal sequences that may vary in speed or length. Originally developed for speech recognition, DTW has since found applications in many fields, such as time series analysis, gesture recognition, and bioinformatics.

#### How DTW Works

- **Alignment**: DTW aligns two sequences by stretching or compressing them to match each other optimally, minimizing the cumulative distance between corresponding points.

- **Warping Path**: It computes the best path through a matrix where each point represents the distance between elements of the two sequences. The path is chosen to minimize the total cost of aligning the sequences.

- **Flexibility**: Unlike other distance measures like Euclidean distance, DTW can handle sequences with varying lengths, non-linear alignments, and temporal distortions.

#### Key Aspects

- **Robust to Time Variations**: DTW can compare sequences that vary in time, making it ideal for applications where the timing of events differs.

- **Versatile Applications**: Beyond its original application (speech and audio), DTW is utilized in finance, robotics, medicine, and any field where pattern recognition in time-dependent data is crucial.

## **Triplet Romanenkova**

This model is based on a **Triplet Network**, which is a type of neural network architecture commonly used for learning embeddings and measuring similarities between data points, particularly in tasks like face recognition, image retrieval, and other metric learning problems. The core idea is to train the network to distinguish between similar and dissimilar pairs using triplet loss. Here, the network was a ResNet.

### Core Components of a Triplet Network

- **Anchor, Positive, and Negative Samples**: The network uses three inputs:
  - **Anchor**: The reference input sample.
      - In this case, the anchor was a random well sequence.
  - **Positive**: A sample similar to the anchor.
      - The positive sample was a sequence close to the anchor in the same well.
  - **Negative**: A sample dissimilar to the anchor.
      - The negative was a random sequence from a different well

- **Shared Weights**: A single neural network is used to process all three inputs with shared weights, ensuring that the learned representations are consistent across samples.

- **Triplet Loss Function**: The key to the Triplet Network is the triplet loss, which ensures that the distance between the anchor and the positive is smaller than the distance between the anchor and the negative by a margin. This loss is formulated as:

  ![Triplet Loss](imgs/triplet_loss.png)

  where \( f(x) \) represents the embedding of input \( x \), and \( \alpha \) is a margin that defines how much closer the positive should be compared to the negative.

#### Key Aspects

- **Learning Discriminative Features**: By directly optimizing the relative distances between samples, the network learns highly discriminative embeddings suitable for similarity and classification tasks.

- **Robust to Variations**: Triplet Networks are effective at handling variations within classes and can generalize well to unseen data by learning more refined boundaries between classes.


## **Variational Autoencoder (VAE)**

**Variational Autoencoder (VAE)** uses a type of autoassociative self-supervised learning that combines neural networks with probabilistic graphical models, allowing it to learn complex data distributions and even generate new samples. VAEs are widely used in applications like image generation, anomaly detection, and representation learning.

### Core Components of a VAE

- **Encoder**: The encoder network maps input data to a latent space by estimating the parameters (mean and variance) of a Gaussian distribution. This probabilistic approach allows the model to learn a smooth latent representation.

- **Latent Space**: Instead of directly encoding inputs, VAEs encode data into a latent space defined by distributions, capturing the underlying factors of the data more effectively.

- **Decoder**: The decoder network reconstructs data from the latent space back into the original data space, generating new samples that resemble the training data.

- **Reparameterization Trick**: A key innovation in VAEs, this trick allows backpropagation through the stochastic layer by separating the randomness from the network parameters, enabling the training of the encoder and decoder using standard gradient descent.

### Loss Function

- **Reconstruction Loss**: Measures how well the generated data matches the original input, typically using mean squared error or binary cross-entropy.

- **KL Divergence**: Regularizes the latent space by forcing the learned distribution to be close to a standard Gaussian distribution, promoting a well-structured latent space.

- **Combined Loss**: The overall VAE loss is the sum of the reconstruction loss and the KL divergence.

### Key Aspects

- **Probabilistic Latent Representation**: VAEs learn a distribution over the latent space, allowing for smooth interpolations and variations in generated samples.

- **Generative Capabilities**: VAEs can generate new, unseen data points by sampling from the learned latent distribution, making them powerful for creative tasks.

- **Applications**: Used in image and speech generation, anomaly detection, and unsupervised learning tasks, VAEs offer flexible and powerful generative modeling.


## **WellGT**

This model is based on a **Triplet Network**, which is a type of neural network architecture commonly used for learning embeddings and measuring similarities between data points, particularly in tasks like face recognition, image retrieval, and other metric learning problems. The core idea is to train the network to distinguish between similar and dissimilar pairs using triplet loss. Here, the network was a ResNet.

However, for the positive and negative sampling, we applied a novel approach. 

The positive sampling was done through a certified methodology for computer vision: augmentation. 
The negative sampling was done through a **BiGAN (Bidirectional Generative Adversarial Network)**. This model extends the standard GAN by jointly learning both the generative model and an inference model (Encoder) that maps data back to the latent space. This bidirectional architecture enables not only the generation of data from latent codes but also the reconstruction of latent codes from data. Using that, we selected random latent vectors in the space that were distant enough from the latent vector of the anchor to generate the negative sequence.

### Core Components of a Triplet Network

- **Anchor, Positive, and Negative Samples**: The network uses three inputs:
  - **Anchor**: The reference input sample.
      - In this case, the anchor was a random well sequence.
  - **Positive**: A sample similar to the anchor.
      - The positive sampling was done through a certified methodology for computer vision: augmentation. 
  - **Negative**: A sample dissimilar to the anchor.
      - The negative sampling was done through a **BiGAN (Bidirectional Generative Adversarial Network)**. This model extends the standard GAN by jointly learning both the generative model and an inference model (Encoder) that maps data back to the latent space. This bidirectional architecture enables not only the generation of data from latent codes but also the reconstruction of latent codes from data. Using that, we selected random latent vectors in the space that were distant enough from the latent vector of the anchor to generate the negative sequence.

- **Shared Weights**: A single neural network is used to process all three inputs with shared weights, ensuring that the learned representations are consistent across samples.

- **Triplet Loss Function**: The key to the Triplet Network is the triplet loss, which ensures that the distance between the anchor and the positive is smaller than the distance between the anchor and the negative by a margin. This loss is formulated as:

  ![Triplet Loss](imgs/triplet_loss.png)

  where \( f(x) \) represents the embedding of input \( x \), and \( \alpha \) is a margin that defines how much closer the positive should be compared to the negative. The \( \alpha \) here was also changed during training. The training starts with a higher \( \alpha \), meaning the distance from anchor-negative should be relatively higher than anchor-positive. The reason was that the data generation from the GAN in the beginning is not good. As the training passes, we lowered the \( \alpha \) so that the distance from anchor-negative should be smaller, since data generation starts getting better.

#### Key Aspects

- **Learning Discriminative Features**: By directly optimizing the relative distances between samples, the network learns highly discriminative embeddings suitable for similarity and classification tasks.

- **Robust to Variations**: Triplet Networks are effective at handling variations within classes and can generalize well to unseen data by learning more refined boundaries between classes.