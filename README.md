# Multi-Task Face Analysis: Age Estimation & Gender Classification
---
## 📌 Project Overview

This project implements a **Multi-Task Learning (MTL)** framework using PyTorch to predict two distinct target variables from a single face image simultaneously:

1. **Age:** A continuous variable (Regression).  
2. **Gender:** A binary variable (Classification).

The goal was to optimize a single model to handle competing objectives using a weighted loss function, progressing from a scratch CNN to a Transfer Learning approach using ResNet34.

## 🧪 Methodology & Model Evolution

The project followed an iterative experimental approach to improve performance and training stability.

### 1. Baseline: Scratch CNN
- **Architecture:** A simple 3-layer CNN with ReLU activations and Max Pooling.  
- **Input:** Resized to `128 × 128`.  
- **Outcome:** High instability during training; the MSE loss (Age) dominated the gradient, leading to poor gender classification performance.

### 2. Improved Custom CNN (v2)
- **Enhancements:** Increased depth to 4 layers and introduced **Batch Normalization** and **Dropout (0.5)** for regularization.  
- **Loss Balancing:** Implemented a weighted multi-task loss function to balance the regression and classification magnitudes:  
  \[
  L_{total} = L_{MSE}(Age) + \lambda \cdot L_{CE}(Gender)
  \]  
  - *Optimal λ:* 50.0

### 3. Transfer Learning (ResNet34) – **Final Model**
- **Backbone:** ResNet34 pretrained on ImageNet.  
- **Heads:** Replaced the final fully connected layer with two parallel heads: one for scalar regression (Age) and one for binary classification (Gender).  
- **Training Strategy:**  
  **2-Phase Fine-Tuning**
  1. **Phase 1:** Freeze backbone, train only heads for 5 epochs (LR = 1e-4).  
  2. **Phase 2:** Unfreeze full model, fine-tune for 13 epochs (LR = 1e-5) with Early Stopping.

---
## 📊 Results

The evaluation metric was the Harmonic Mean of Age **nRMSE** and Gender **Macro F1 Score**.

| Model | Description | Age RMSE (↓) | Gender F1 (↑) |
|-------|-------------|--------------|----------------|
| **Model 1** | Scratch CNN (Baseline) | 128.39 | 0.8784 |
| **Model 2** | Scratch CNN v2 + Balanced Loss | 10.93 | 0.8618 |
| **Model 3** | **ResNet34 (Transfer Learning)** | **8.46** | **0.9157** |

---
## 🛠️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/multi-task-face-prediction.git
```
### 2.Install Dependencies
```bash
pip install torch torchvision pandas numpy scikit-learn pillow
```
### 3. Run the Notebook
Open `Code_file.ipynb` to view the data processing, training loop, and inference pipeline.

---
## 🚀 Future Work (Suggested)

1. **Ordinal Regression for Age**  
   Treat age as an ordered category (e.g., CORAL, ordinal bins) to improve robustness and reduce outlier sensitivity.
2. **Attention Mechanisms**  
   Integrate CBAM or self-attention modules to highlight facial regions influencing predictions.
3. **Uncertainty Estimation**  
   Use Monte Carlo Dropout or Deep Ensembles to measure confidence in age and gender predictions.
4. **Fairness Analysis**  
   Evaluate performance across demographic subgroups to detect and reduce potential model bias.
5. **Advanced Multi-Task Optimization**  
   Experiment with methods like **GradNorm**, **PCGrad**, or **Dynamic Weight Averaging** to balance competing task gradients.
6. **Stronger Backbones**  
   Explore larger or more modern architectures such as ResNet50, MobileNetV3, or ViT-based multi-head models.

