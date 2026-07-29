# Transfer Learning for Cats vs Dogs Classification

## Overview

This project implements transfer learning for binary image classification using a pre-trained MobileNetV2 model. The approach involves two training stages: feature extraction with a frozen base model, followed by fine-tuning of selected layers with a reduced learning rate.

### Motivation for Transfer Learning

Transfer learning addresses the challenge of limited training data by leveraging knowledge learned from large-scale datasets. Rather than training a deep neural network from scratch, which requires substantial computational resources and large amounts of labeled data, transfer learning reuses pre-trained models. MobileNetV2, trained on ImageNet with 1.4 million images across 1,000 classes, has already learned robust low-level and mid-level features (edges, textures, shapes) that are transferable to the cats vs dogs classification task. This approach significantly reduces training time and improves generalization with limited data.

## Dataset

The project uses the `cats_vs_dogs` dataset from TensorFlow Datasets. After removing 1,738 corrupted images, 23,262 valid images remain. The dataset is split into:
- Training set: 80% (approximately 18,610 images)
- Validation set: 20% (approximately 4,652 images)

All images are resized to 224×224 pixels to match the MobileNetV2 input requirements.

## Model Architecture

### Base Model
MobileNetV2 pre-trained on ImageNet serves as the feature extractor. The original classification head is removed (`include_top=False`), retaining only the convolutional base with approximately 3.5 million parameters.

### Custom Head
A new classification head is added on top of the frozen base:
- GlobalAveragePooling2D: Converts feature maps to a 1D vector
- Dropout (0.2): Regularization to prevent overfitting
- Dense layer (1 unit, sigmoid activation): Binary classification output

## Training Procedure

### Stage 1: Feature Extraction
- All MobileNetV2 layers are frozen
- Only the new classification head is trained
- Learning rate: 1×10⁻³
- Optimizer: Adam
- Loss: Binary crossentropy
- Epochs: 3
- Batch size: 32

**Rationale**: In this stage, the pre-trained convolutional base acts as a fixed feature extractor. By freezing all weights, we preserve the learned representations from ImageNet while training only the task-specific classifier. This approach is computationally efficient and prevents catastrophic forgetting of the pre-trained features. The higher learning rate (1×10⁻³) is appropriate since the classification head is randomly initialized.

### Stage 2: Fine-tuning
- The last 100 layers of MobileNetV2 are unfrozen
- Training continues with a reduced learning rate: 1×10⁻⁵
- Optimizer: Adam
- Loss: Binary crossentropy
- Epochs: 3
- Batch size: 32

**Rationale**: Fine-tuning allows the model to adapt higher-level features to the specific characteristics of the cats vs dogs dataset. By unfreezing the last 100 layers (approximately the top 30% of the network), we enable selective adaptation of features while preserving the robustness of lower-level features. The significantly reduced learning rate (1×10⁻⁵, which is 100 times lower than Stage 1) is critical to prevent large weight updates that could damage the pre-trained representations. This conservative approach balances adaptation with stability.

## Data Preprocessing

Images undergo the following preprocessing steps:
1. Resize to 224×224 pixels
2. Normalize pixel values using MobileNetV2's preprocessing function
3. Data augmentation (training set only):
   - Random horizontal flip
   - Random rotation (±10 degrees)
   - Random zoom (±10%)

## Requirements

- Python 3.10 or newer
- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib
- NumPy

### System Requirements

- **RAM**: Minimum 4 GB (8 GB recommended for comfortable training)
- **GPU**: Optional but highly recommended (NVIDIA CUDA-compatible GPU reduces training time by 5-10×)
- **Disk Space**: Approximately 1 GB for dataset download and model storage

Install dependencies:
```bash
pip install tensorflow tensorflow-datasets matplotlib numpy
```

## Usage

1. Open `final_assignment.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially

The notebook performs the following operations:
- Downloads and loads the cats_vs_dogs dataset
- Preprocesses and augments the data
- Loads MobileNetV2 and adds a custom classification head
- Trains the model in two stages
- Evaluates performance on the validation set
- Saves the trained model as `cats_dogs_finetuned.h5`

## Results

The model achieves high accuracy after the two-stage training process. Training and validation accuracy curves are plotted to visualize convergence. The final model is saved in HDF5 format for future inference.

### Expected Performance

- **Stage 1 Accuracy**: Approximately 85-90% on validation set
- **Stage 2 Accuracy**: Approximately 92-96% on validation set
- **Training Time**: 10-30 minutes on GPU; 1-3 hours on CPU
- **Model Size**: Approximately 35 MB (HDF5 format)
- **Inference Time**: Approximately 50-100 ms per image on GPU

### Performance Factors

The actual performance depends on several factors:
- Hardware acceleration (GPU vs CPU)
- Random seed initialization
- Exact number of training iterations
- Data augmentation effects
- Batch normalization statistics

## References

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://arxiv.org/abs/1801.04381

- Yosinski, J., Clune, J., Bengio, Y., & Liphardt, H. (2014). How transferable are features in deep neural networks? In Advances in Neural Information Processing Systems (NIPS). https://arxiv.org/abs/1411.1792

- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359. https://doi.org/10.1109/TKDE.2009.191

- Keras Transfer Learning Guide: https://keras.io/guides/transfer_learning/

- TensorFlow Datasets - cats_vs_dogs: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs

- MobileNetV2 Architecture: https://keras.io/api/applications/mobilenet/

## Notes

- The dataset is automatically downloaded on first run and cached for subsequent runs
- Training time varies depending on hardware (GPU recommended)
- The saved model can be loaded using `keras.models.load_model('cats_dogs_finetuned.h5')`
- Memory consumption during training: approximately 2-4 GB on GPU
- The model weights are stored in HDF5 format, compatible with TensorFlow 2.x and later versions
