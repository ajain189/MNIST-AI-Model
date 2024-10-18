# Handwritten Digit Classification using ResNet-18

This project classifies handwritten digits (0-9) from the MNIST dataset using a pretrained ResNet-18 model in PyTorch. The model is trained to achieve high accuracy and can be used to classify digits from both the training and test datasets.

## Project Structure
```
├── data/                  # MNIST dataset
│   ├── train/             # Training images
│   ├── test/              # Testing images
├── model/                 # Fully trained ResNet-18 model
├── train_mnist.py         # Script to train the model
├── test_mnist.py          # Script to test the model on test data
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Features
- **Pretrained ResNet-18**: Utilizes the ResNet-18 architecture for digit classification.
- **Dataset Handling**: MNIST dataset is loaded using PyTorch's `Dataset` and `DataLoader`.
- **Data Transformations**: Applies transformations such as normalization to prepare the dataset.
- **Training and Testing**: Separate scripts for training (`train.py`) and testing (`test.py`) the model.
- **Optimizer**: Uses the Adam optimizer.
- **Loss Function**: Cross Entropy Loss is used to measure the performance of the model.
- **Final Performance**: Achieved a total loss of 0.08%.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib

You can install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage

### Training
To train the model, run the following command:
```bash
python train_mnist.py
```
The training script reads the dataset from the `data/train/` directory and trains the model using the ResNet-18 architecture. The trained model will be saved in the `model/` directory.

### Testing
To test the model's accuracy on the test dataset, run:
```bash
python test_mnist.py
```
The script will load the pretrained model from `model/` and evaluate it on the `data/test/` dataset, printing the accuracy and other evaluation metrics.

## Results
- **Final Loss**: 0.08%
- **Model**: Pretrained ResNet-18 fine-tuned on MNIST
- **Evaluation Metrics**: Accuracy, loss, and confusion matrix can be generated after testing.

## Future Improvements
- Implement additional data augmentation techniques for improved generalization.
- Experiment with different optimizers and learning rate schedules.
- Expand the project to other datasets or tasks using similar architectures.
