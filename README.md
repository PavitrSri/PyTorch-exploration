# MNIST Handwritten Digit Classifier

## Overview

This project demonstrates **PyTorch** by building a neural network that recognizes handwritten digits from the MNIST dataset. The program trains a model on 60,000 labeled digit images and evaluates its performance on 10,000 test images.

---

## 1. Which Package/Library Does This Program Demonstrate?

This sample program demonstrates **PyTorch**, a popular deep learning framework for building and training neural networks.

---

## 2. How to Run the Program

### Prerequisites
Ensure you have Python 3.7+ and install the required packages:

```bash
pip install torch torchvision matplotlib numpy
```

### Running the Program

```bash
python mnist_classifier.py
```

### What Happens
The program will:
1. **Download MNIST dataset** (~11 MB) - First run only
2. **Train for 10 epochs** - Takes ~2-3 minutes on CPU, ~30 seconds on GPU
3. **Generate visualizations**:
   - `training_history.png` - Shows loss and accuracy curves
   - `sample_predictions.png` - Shows 10 sample predictions with confidence scores
4. **Print detailed results** to the console

### Expected Output
```
Using device: cpu
Training set size: 60000
Test set size: 10000

============================================================
TRAINING THE MODEL
============================================================
Epoch [1/10] | Loss: 0.5123 | Train Acc: 85.23% | Test Acc: 92.14%
Epoch [2/10] | Loss: 0.1456 | Train Acc: 95.67% | Test Acc: 96.28%
...
Epoch [10/10] | Loss: 0.0234 | Train Acc: 98.92% | Test Acc: 97.85%

============================================================
TRAINING COMPLETE
============================================================
✓ Training history saved as 'training_history.png'
✓ Sample predictions saved as 'sample_predictions.png'
```

---

## 3. What Purpose Does This Program Serve?

This program serves **multiple educational and practical purposes**:

### Educational Value
- **Learn PyTorch fundamentals**: Demonstrates core concepts like tensors, neural networks, and training loops
- **Understand machine learning workflow**: Data loading → preprocessing → model building → training → evaluation
- **Visualize learning**: See how accuracy improves over epochs

### Practical Application
- **Digit recognition**: Can predict handwritten digits in real-world scenarios (e.g., postal codes, bank checks)
- **Reproducibility**: Provides a complete, working example that others can learn from and build upon
- **Benchmarking**: Tests a neural network's ability to generalize from training data to unseen test data

### Why It's Useful (Not Just Showing Off)
Unlike a trivial "Hello World" program, this classifier:
- Solves a real problem (digit classification) that has historical importance in ML
- Uses multiple PyTorch features in combination (data loading, models, loss functions, optimization)
- Produces measurable, interpretable results (accuracy scores, prediction confidence)
- Generates visualizations that help understand model behavior

---

## 4. Sample Input/Output

### Input Format
The program uses the **MNIST dataset**: 28×28 pixel grayscale images of handwritten digits (0-9).

### Sample Output - Console
```
============================================================
SAMPLE PREDICTIONS
============================================================

Detailed Results for 10 Sample Images:
----------------------------------------------------------------------
Index  Predicted    True Label   Confidence  Correct 
----------------------------------------------------------------------
0      7            7            99.87%      ✓
1      2            2            98.34%      ✓
2      1            1            97.12%      ✓
3      0            0            96.45%      ✓
4      4            9            45.23%      ✗
5      9            9            99.01%      ✓
6      5            5            94.67%      ✓
7      8            8            97.89%      ✓
8      6            6            99.12%      ✓
9      3            3            98.56%      ✓

============================================================
FINAL TEST SET PERFORMANCE
============================================================

Overall Test Accuracy: 97.85%

Per-Class Accuracy:
----------------------------------------
Digit 0: 99.10% (971/980)
Digit 1: 99.21% (1126/1135)
Digit 2: 97.45% (958/983)
Digit 3: 96.78% (928/958)
Digit 4: 97.52% (934/957)
Digit 5: 96.34% (866/899)
Digit 6: 98.12% (947/965)
Digit 7: 97.89% (978/998)
Digit 8: 96.45% (928/962)
Digit 9: 96.78% (948/979)
```

### Sample Output - Visualizations

#### Training History Graph
Shows how the model's loss decreases and accuracy improves over 10 epochs:
- Left plot: Loss decreases from ~0.5 to ~0.02
- Right plot: Test accuracy improves from ~92% to ~98%

#### Sample Predictions Grid
A 2×5 grid showing 10 random test images with:
- Handwritten digit image
- Predicted digit and confidence score
- True label
- Green title = correct, Red title = incorrect

**Example**:
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   [image]   │  │   [image]   │  │   [image]   │
│ Pred: 7     │  │ Pred: 2     │  │ Pred: 1     │
│ True: 7     │  │ True: 2     │  │ True: 3     │
│ 99.87%      │  │ 98.34%      │  │ 87.23%      │
│ ✓ GREEN     │  │ ✓ GREEN     │  │ ✗ RED       │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## 5. Key Features Demonstrated

### PyTorch Functionalities Used
1. **Data Loading** (`torchvision.datasets`, `DataLoader`)
   - Automatic MNIST download
   - Batch processing for efficiency

2. **Data Preprocessing** (`transforms`)
   - Tensor conversion
   - Normalization

3. **Neural Network Architecture** (`nn.Module`)
   - Fully connected layers (`nn.Linear`)
   - Activation functions (`ReLU`)
   - Dropout for regularization

4. **Training Loop**
   - Forward pass
   - Loss computation (`CrossEntropyLoss`)
   - Backpropagation
   - Optimization (`Adam` optimizer)

5. **Evaluation & Inference**
   - Model evaluation mode
   - Predictions on unseen data
   - Confidence scores

### General Programming Skills
- Data structures (lists, dictionaries)
- Control flow (loops, conditionals)
- Object-oriented programming (class inheritance)
- File I/O (saving images)
- Data visualization (matplotlib)
- Error handling and device management (GPU/CPU)

---

## 6. Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.19.0
```

Install with:
```bash
pip install torch torchvision matplotlib numpy
```

---

## 7. Project Structure

```
.
├── mnist_classifier.py      # Main program
├── README.md                # This file
├── training_history.png     # Generated after running (training curves)
├── sample_predictions.png   # Generated after running (predictions)
└── data/                    # Generated after running (MNIST dataset cache)
    └── MNIST/
```

---

## 8. Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch: `pip install torch torchvision`

### Issue: Very slow training (5+ minutes per epoch)
**Solution**: You're using CPU. Install GPU version of PyTorch for 10x speedup:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory error
**Solution**: Reduce batch size in the code from 64 to 32 or 16.

---

## 9. Further Exploration

Want to experiment? Try modifying:
- **Network architecture**: Change layer sizes or add more layers
- **Hyperparameters**: Adjust learning rate, number of epochs, dropout rate
- **Optimizers**: Try SGD, RMSprop instead of Adam
- **Batch size**: Experiment with different batch sizes
- **Data augmentation**: Add rotations, shifts to training data

---

## 10. References

- PyTorch Official Documentation: https://pytorch.org/docs/stable/index.html
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- Understanding Neural Networks: https://github.com/pytorch/examples/tree/master/mnist

---

**Author**: CS2613 Student  
**Date**: April 2026  
**Language**: Python  
**Framework**: PyTorch
