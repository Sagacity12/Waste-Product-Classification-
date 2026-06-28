# Waste Product Classification

A deep learning project for classifying waste products into two categories: Organic (O) and Recyclable (R) using transfer learning with VGG16.

## Project Overview

This project implements a binary image classifier to distinguish between organic and recyclable waste products. Two models are trained and compared:

1. **Feature Extraction Model**: Uses VGG16 as a frozen feature extractor with custom classification layers
2. **Fine-Tuned Model**: Fine-tunes the last convolutional block of VGG16 for improved performance

## Dataset

- **Source**: IBM Cloud Object Storage
- **Total Images**: 1,200
  - Training: 800 images
  - Validation: 200 images
  - Testing: 200 images
- **Classes**:
  - O (Organic waste)
  - R (Recyclable waste)
- **Image Size**: 150x150 pixels

## Model Architecture

### Base Model
- **Backbone**: VGG16 (pre-trained on ImageNet)
- **Input Shape**: (150, 150, 3)
- **Top Layers**: Custom dense layers with dropout

### Feature Extraction Model
- All VGG16 layers frozen
- Flatten layer
- Dense layer (512 units, ReLU)
- Dropout (0.3)
- Dense layer (512 units, ReLU)
- Dropout (0.3)
- Output layer (1 unit, Sigmoid)

### Fine-Tuned Model
- Layers from `block5_conv3` onwards are trainable
- Same custom top layers as feature extraction model

## Training Configuration

- **Optimizer**:
  - Feature Extraction: Adam (lr=1e-5)
  - Fine-Tuning: RMSprop (lr=1e-4)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)
- **Learning Rate Schedule**: Exponential decay
- **Data Augmentation**:
  - Width shift (0.1)
  - Height shift (0.1)
  - Horizontal flip
  - Rescaling (1/255)

## Callbacks

- **Early Stopping**: Monitors validation loss (patience=4, min_delta=0.01)
- **Model Checkpoint**: Saves best model based on validation loss
- **Learning Rate Scheduler**: Exponential decay
- **Custom Loss History**: Tracks loss and learning rate per epoch

## Results

### Feature Extraction Model
- **Test Accuracy**: 77%
- **Precision**: 0.78 (macro avg)
- **Recall**: 0.77 (macro avg)
- **F1-Score**: 0.77 (macro avg)

### Fine-Tuned Model
- **Test Accuracy**: 82%
- **Precision**: 0.83 (macro avg)
- **Recall**: 0.82 (macro avg)
- **F1-Score**: 0.82 (macro avg)

The fine-tuned model shows a **5% improvement** in accuracy over the feature extraction model.

## Requirements

```
tensorflow>=2.21.0
numpy
matplotlib
scikit-learn
requests
tqdm
```

## Project Structure

```
Waste Product Classification/
│
├── Classify Waste Product.ipynb    # Main notebook
├── README.md                        # Project documentation
├── O_R_tlearn_vgg16.keras          # Saved feature extraction model
├── O_R_tlearn_fine_tune_vgg16.keras # Saved fine-tuned model
└── o-vs-r-split/                    # Dataset directory
    ├── train/
    │   ├── O/                       # Organic waste images
    │   └── R/                       # Recyclable waste images
    ├── test/
    │   ├── O/
    │   └── R/
    └── val/
        ├── O/
        └── R/
```

## Usage

1. **Run the notebook**: Open `Classify Waste Product.ipynb` in Jupyter Notebook/Lab
2. **Download dataset**: The notebook automatically downloads and extracts the dataset
3. **Train models**: Execute cells sequentially to train both models
4. **Evaluate**: View training curves and classification reports
5. **Make predictions**: Use saved models to classify new images

## Key Features

- Transfer learning with VGG16
- Comparison of feature extraction vs fine-tuning approaches
- Comprehensive data augmentation
- Learning rate scheduling
- Model checkpointing
- Visualization of training metrics
- Detailed performance evaluation

## Future Improvements

- Multi-class classification (more waste categories)
- Implement other pre-trained models (ResNet, EfficientNet)
- Deploy as web application
- Real-time classification from camera feed
- Increase dataset size for better generalization

## License

This project is open-source and available for educational purposes.

## Acknowledgments

- Dataset provided by IBM Cloud Object Storage
- VGG16 model pre-trained on ImageNet
- TensorFlow/Keras framework

## AI Assistant
git push origin main --force
