# Early-stage-disease-detection
Final year project where we detect the disease spread over the plant leaf image using data augmentation and Deep learning models.

# Early Stage Crop Disease Prediction using Novel Augmentation Methods

**Status:** `In Progress` | **Current Milestone:** `M1 - Panel Review (30%)`

## 1. Project Abstract

This project aims to develop a robust system for the early-stage detection and classification of diseases in agricultural crops like tomato, potato, and corn. The core innovation of this research lies in the development and application of 12 novel, biologically-informed data augmentation techniques. These methods are designed to create a more realistic and diverse training dataset that specifically addresses the challenges of subtle, early-stage symptoms, which are often underrepresented in standard datasets and nearly invisible to the human eye. The final model will be deployed on a web-based platform for easy use.

## 2. The Problem Statement

Deep learning models for plant disease detection are highly dependent on the quality and diversity of training data. Standard datasets often lack sufficient examples of diseases in their earliest stages. Consequently, models trained on this data may fail to generalize to real-world conditions where lighting, background, leaf orientation, and the gradual progression of a disease are highly variable. This project directly tackles this data scarcity problem.

## 3. Our Novel Solution

We have devised a suite of 12 hybrid data augmentation methods that go beyond simple geometric and color transformations. These techniques intelligently manipulate image data based on the context of plant biology and disease pathology.

For the first milestone, we are implementing and evaluating:
* **SPLA (Symptom-Preserving Local Augmentation):** Modifies the healthy parts of a leaf (background, lighting, texture) while keeping the subtle disease lesions completely unchanged, forcing the model to become robust to environmental variations.
* **SIM (Stage-Interpolation Morphing):** Creates synthetic "in-between" images by morphing and blending images of a disease at different stages (e.g., early and mid-stage). This teaches the model about the gradual progression of symptoms.

## 4. Project Structure

The project follows a standard machine learning project structure for better organization and collaboration.

```
├── data/
│   ├── raw/          # Original PlantVillage dataset
│   └── processed/    # Train/Validation/Test splits
├── notebooks/
│   └── results_analysis.ipynb # For visualizing results and generating plots
├── src/
│   ├── augmentations.py  # All novel augmentation functions (SPLA, SIM, etc.)
│   ├── train.py          # Master script for training models
│   └── utils.py          # Helper functions (e.g., data loading, plotting)
├── results/
│   ├── models/         # Saved model weights (.h5 or .pth files)
│   └── plots/          # Saved confusion matrices, Grad-CAM visualizations
└── README.md
```

## 5. Getting Started: Setup and Installation

Follow these steps to set up the project environment on your local machine.

### Prerequisites
* Python 3.8+
* Git

### Procedure

**1. Clone the repository:**
```bash
git clone [your-github-repository-url]
cd [your-project-directory]
```

**2. Create a Python Virtual Environment:**
* **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
* **On macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**3. Install Dependencies:**
We use a `requirements.txt` file to manage all project dependencies.
```bash
pip install -r requirements.txt
```
*(Note: Your team needs to create this file by running `pip freeze > requirements.txt` after installing the necessary libraries).* Key libraries include: `tensorflow`, `opencv-python`, `scikit-learn`, `matplotlib`, `pandas`, `seaborn`.

**4. Download the Dataset:**
* Download the **PlantVillage dataset**.
* Unzip and place the relevant crop folders (e.g., `Tomato___...`) into the `data/raw/` directory.
* Run the data preprocessing script (or manually create) to split the data into `train`, `validation`, and `test` sets inside the `data/processed/` directory.

## 6. How to Run the Project

The primary script for all experiments is `src/train.py`.

### Training a Model

You can train different models by passing command-line arguments.

**1. Train the Baseline Model:**
This model is trained on the original, un-augmented data.
```bash
python src/train.py --model_type baseline --epochs 15
```

**2. Train with SPLA Augmentation:**
This will apply the SPLA function on-the-fly to the training data.
```bash
python src/train.py --model_type spla --epochs 25
```

**3. Train with SIM Augmentation:**
This will apply the SIM function on-the-fly to the training data.
```bash
python src/train.py --model_type sim --epochs 25
```
*Models and training history will be saved in the `results/` directory.*

### Analyzing Results

All results, comparisons, and visualizations are generated in the `notebooks/results_analysis.ipynb` Jupyter Notebook. Open and run the cells in the notebook to see the final comparison tables, confusion matrices, and Grad-CAM visualizations.

## 7. Contribution Guidelines

To ensure a smooth workflow and avoid conflicts, please follow these Git practices:

1.  **Create a new branch** for every new feature or task (e.g., `git checkout -b feat/implement-spla`).
2.  **Commit your changes** with clear, descriptive messages.
3.  **Push your branch** to the remote repository (`git push origin feat/implement-spla`).
4.  **Open a Pull Request (PR)** on GitHub, assigning a teammate to review your code.
5.  Once the PR is approved and passes checks, it can be merged into the `main` branch.
6.  **NEVER** push directly to the `main` branch.

## 8. Team Members
* [Your Name]
* [Teammate 1's Name]
* [Teammate 2's Name]

---