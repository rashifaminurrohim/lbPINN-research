# lbPINN for Crop Classification

> Integrating agronomic domain knowledge into neural network classification using Physics-Informed Neural Network with Uncertainty-Based Loss Balancing (lbPINN)

---

## Overview

This repository contains the implementation of my undergraduate thesis research comparing two Physics-Informed Neural Network approaches for crop recommendation classification:

- **Fixed-weight PINN** — integrates agronomic constraints with a fixed penalty weight (λ=1.0)
- **lbPINN** (loss-balanced PINN) — uses uncertainty-based adaptive weighting to automatically balance data loss and physics loss during training

The physics constraints are derived from the **ECOCROP database (FAO)**, which provides optimal temperature and pH ranges for 22 crop types. These constraints are transformed into Gaussian parameters (μ, σ) and integrated directly into the loss function.

---

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | Misclassifications |
|---|---|---|---|---|---|
| MLP Baseline | 99.39% | 99.43% | 99.39% | 99.39% | 2/330 |
| Fixed-weight PINN | 98.48% | 98.61% | 98.48% | 98.48% | 5/330 |
| **lbPINN** | **99.09%** | **99.18%** | **99.09%** | **99.09%** | **3/330** |

**lbPINN adaptive weighting at epoch 100:**
- ε_data: 1.000 → 0.172 (effective weight: 0.5 → **16.88**)
- ε_physics: 1.000 → 0.929 (effective weight: 0.5 → **0.58**)

---

## Physics Loss Formulation

The physics loss in this research uses a two-component max-based penalty:

$$\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^{N} \max\left((1 - crop_{prob,i}) \times env\_score_i,\ crop_{prob,i} \times (1 - env\_score_i)\right)$$

This captures two directions of misalignment:
- **Component 1** — model is not confident despite favorable environmental conditions
- **Component 2** — model is overconfident despite unfavorable environmental conditions

The environmental suitability score is computed using Gaussian functions over temperature and pH:

$$env\_score = \exp\left(-\frac{(T_{input}-\mu_T)^2}{2\sigma_T^2}\right) \times \exp\left(-\frac{(pH_{input}-\mu_{pH})^2}{2\sigma_{pH}^2}\right)$$

---

## Dataset

- **Crop Recommendation Dataset** — 2,200 samples, 22 crop classes, 7 features (N, P, K, temperature, humidity, pH, rainfall), perfectly balanced (100 samples/class)
- **ECOCROP (FAO)** — optimal temperature and pH ranges for each crop, transformed to Gaussian parameters

---

## Model Architecture

```
Input (7) → Dense (32, ReLU) → Output (22, Softmax)
Total trainable parameters: 982
```

---

## Repository Structure

```
├── 00_preprocessing.ipynb   # Label encoding, StandardScaler, train/val/test split, ECOCROP transformation
├── 01_baseline.ipynb        # MLP baseline (data loss only)
├── 02_fixed_pinn.ipynb      # Fixed-weight PINN (λ=1.0)
├── 03_lbpinn.ipynb          # lbPINN with uncertainty-based adaptive weighting
├── data/
│   ├── Crop_Recommendation.csv
│   └── constraints_minmax.json   # ECOCROP optimal ranges (temperature, pH)
└── outputs/                 # Saved models, session variables, figures
```

---

## How to Run

**Requirements:**
```
tensorflow>=2.10
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
```

**Run order:**
```
1. 00_preprocessing.ipynb   → generates session_preprocessing.pkl
2. 01_baseline.ipynb        → optional, baseline reference
3. 02_fixed_pinn.ipynb      → Fixed-weight PINN training & evaluation
4. 03_lbpinn.ipynb          → lbPINN training & evaluation
```

Each notebook loads the preprocessed data from `session_preprocessing.pkl` — run preprocessing first, then any model notebook independently.

---

## lbPINN Total Loss Formulation

$$\mathcal{L}_{total} = \frac{1}{2\epsilon_{data}^2}\mathcal{L}_{data} + \frac{1}{2\epsilon_{physics}^2}\mathcal{L}_{physics} + \log(\epsilon_{data} \cdot \epsilon_{physics})$$

where ε_data and ε_physics are trainable uncertainty parameters optimized simultaneously with network weights via gradient descent.

---

## References

- Latha & Kumaresan (2025). Crop Recommendation Dataset. *Mendeley Data*
- The State of Food and Agriculture 2021. (2021). In The State of Food and Agriculture 2021. FAO. https://doi.org/10.4060/cb4476en
- Gunasekaran, H., Ramalakshmi, K., Debnath, S., & Swaminathan, D. K. (2025). Physics-Aware Ensemble Learning for Superior Crop Recommendation in Smart Agriculture. Sensors, 25(19). https://doi.org/10.3390/s25196243
- Xiang, Z., Peng, W., Zheng, X., Zhao, X., & Yao, W. (2022). Self-adaptive loss balanced Physics-informed neural networks. Neurocomputing, 496(1), 11–34. https://doi.org/10.1016/j.neucom.2022.05.015
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*
- Kendall, A., & Gal, Y. (2018). Multi-task learning using uncertainty to weigh losses. *CVPR*

---

## Author

**Muhammad Rashif Aminurrohim**

Informatics — Institut Teknologi Nasional Bandung
