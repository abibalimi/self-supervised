# Self-Supervised Learning


## Constrastive Learning: 

### Implementation Steps

- Data Augmentation: Create two augmented views of each image.
- Encoder Network: Use a convolutional neural network (CNN) to encode the images into feature vectors.
- Projection Head: Map the feature vectors to a lower-dimensional space where contrastive loss is applied.
- Contrastive Loss: Use the NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss to bring positive pairs closer and push negative pairs apart.
