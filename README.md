
# Brain Tumor Classification using Joint Graph Embedding Knowledge Distillation (JGEKD)

### 🧠 Overview

This project applies **Joint Graph Embedding Knowledge Distillation (JGEKD)** to classify **brain tumors from MRI scans**.
It distills knowledge from a **high-capacity teacher model (ResNet50)** into a **compact student model (ResNet18)** using a **graph-based relational loss** to preserve inter-class feature similarity.

The result is a lightweight model that maintains high accuracy while remaining computationally efficient, ideal for **edge deployment** and **real-time medical diagnostics**.

---

### 🚀 Key Highlights

* Implements **graph-based knowledge distillation (JGEKD)** for structured knowledge transfer
* Combines **CrossEntropy Loss** and **JGEKD Loss** for balanced learning
* Handles **class imbalance** through dynamic dataset balancing
* Includes **training visualizations** and **confusion matrix** generation
* Optimized for **MPS (Apple Silicon)**, **CUDA**, and **CPU** execution
* Compatible with the **Kaggle Brain Tumor MRI Dataset**

---

### 📂 Dataset Structure

Your dataset should follow this directory format:

```
classification_task/
│
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

Each subfolder should contain `.jpg`, `.png`, or `.jpeg` MRI images.

**Recommended Dataset:**
[Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

---

### ⚙️ Installation

Install all dependencies:

```
pip install pandas numpy pillow matplotlib seaborn scikit-learn torch torchvision kagglehub
```

---

### 🧩 Project Architecture

| Component                | Description                                                                    |
| ------------------------ | ------------------------------------------------------------------------------ |
| **JGEKDLoss**            | Custom graph-based knowledge distillation loss preserving relational structure |
| **BrainTumorDataset**    | Loads and preprocesses MRI scans using PyTorch transforms                      |
| **get_data_loaders()**   | Splits and balances training and testing sets                                  |
| **train_and_validate()** | Trains model using CE and KD losses, logs progress                             |
| **evaluate_metrics()**   | Generates confusion matrix and classification report                           |
| **plot_metrics()**       | Visualizes training loss and accuracy across epochs                            |

---

### 🧠 Model Setup

| Role        | Architecture | Params | Pretrained | Trainable |
| ----------- | ------------ | ------ | ---------- | --------- |
| **Teacher** | ResNet50     | 25.6M  | ImageNet   | Frozen    |
| **Student** | ResNet18     | 11.7M  | ImageNet   | Trainable |

**Loss Functions:**

* CrossEntropy Loss for label supervision
* JGEKDLoss for relational knowledge transfer

**Optimizer:** Adam
**Learning Rate:** 1e-4
**Batch Size:** 32
**Epochs:** 10
**Lambda (KD weight):** 0.5

---

### 🧮 Training Example

```python
teacher_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, NUM_CLASSES)
teacher_model.eval()

student_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
student_model.fc = nn.Linear(student_model.fc.in_features, NUM_CLASSES)

ce_loss_fn = nn.CrossEntropyLoss()
kd_loss_fn = JGEKDLoss()
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

train_losses, test_accuracies = train_and_validate(
    student_model, teacher_model, train_loader, test_loader,
    optimizer, ce_loss_fn, kd_loss_fn, lambda_weight=0.5, epochs=10, device=DEVICE
)
```

---

### 📊 Training Output

```
Epoch 1/10, Train Loss: 0.8931, Test Accuracy: 84.12%
Epoch 2/10, Train Loss: 0.7145, Test Accuracy: 88.76%
Epoch 3/10, Train Loss: 0.5421, Test Accuracy: 92.03%
...
```

**Visual Metrics:**

* Loss vs Epoch
* Accuracy vs Epoch
* Confusion Matrix
* Precision, Recall, and F1 Report

---

### 🧾 Evaluation Report

| Class                | Precision | Recall | F1-score |
| -------------------- | --------- | ------ | -------- |
| Glioma               | 0.95      | 0.93   | 0.94     |
| Meningioma           | 0.91      | 0.92   | 0.92     |
| No Tumor             | 0.94      | 0.95   | 0.94     |
| Pituitary            | 0.96      | 0.97   | 0.96     |
| **Overall Accuracy** | **94.5%** |        |          |

*(Values are approximate and vary per run.)*

---

### ☁️ Future Enhancements

* Add **AWS S3 or Google Drive** integration for cloud data storage
* Develop a **web dashboard (Flask/FastAPI)** for model inference
* Implement **Grad-CAM** for MRI region visualization
* Build **real-time inference system** for clinical settings
* Extend to **multi-modal fusion (MRI + CT)** for improved diagnosis

---

### 📈 Performance Insights

The **JGEKD framework** allows the student model to:

* Retain relational structure learned by the teacher
* Reduce parameters while maintaining accuracy
* Achieve smoother decision boundaries and improved generalization

---

### 🧑‍💻 Credits

* **Dataset:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
* **Icons & Graphics:** [Flaticon](https://www.flaticon.com/), [PngTree](https://pngtree.com/)
* **Frameworks:** PyTorch, Torchvision, Scikit-learn, Matplotlib, Seaborn

---

### 👨‍🎓 Author

**Abrar Ahmad**
* M.Tech in Artificial Intelligence |
* Email: [abrar.ahmad@iust.ac.in](mailto:abrar.ahmad@iust.ac.in)
* LinkedIn: [linkedin.com/in/siyran-shafi](https://www.linkedin.com/in/siyran-shafi)
* GitHub: [github.com/siyran-shafi](https://github.com/siyran-shafi)

---

### 📜 License

This project is licensed under the **MIT License**.
You may use, modify, and distribute it with appropriate credit to the author.


