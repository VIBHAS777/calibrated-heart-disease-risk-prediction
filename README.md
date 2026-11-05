# Heart Disease Risk Prediction Using Deep Learning and Probabilistic Calibration Techniques

[![Journal](https://img.shields.io/badge/Journal-Engineered_Science_(Q1)-brightgreen.svg)](https://www.espublisher.com/journals/id/es)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Published in ***Engineered Science* (Q1 Journal)**.  
*Manuscript ID: ES-2026-0082.R2*

---

## 👥 Authors
- **Bajjareddy Venkatesh**
- **Rashmi M R**
- **Meera K**
- **Hoong Pine Lee**
- **Hima Sagar B**
- **Araveeti Vibhas Reddy** ([@VIBHAS777](https://github.com/VIBHAS777))
- **Bharghav Vardhan Reddy Irigela**

---

## 🩺 Overview
In life-critical clinical environments, raw machine learning risk estimates cannot be taken at face value. Standard deep neural networks are notoriously overconfident, often assigning 90%+ risk to uncertain cases or failing to calibrate confidence with true patient event frequencies.

This repository provides the official implementation of our calibrated clinical risk architecture:
1. **Bagged Multi-Layer Perceptron (MLP):** Leverages **Bayesian Bootstrap Aggregation** to stabilize variance on clinical biometrics.
2. **Probabilistic Calibration:** Compares **Platt Scaling** (sigmoidal calibration) and **Isotonic Regression** (monotonic non-parametric calibration) to minimize the **Brier Score** to **0.0489**.
3. **Safety Threshold Optimization:** Enforces a critical false negative rate of **< 3%**, guaranteeing that at-risk cardiac patients are not misclassified as healthy.

---

## 📁 Repository Structure
```
calibrated-heart-disease-risk-prediction/
├── paper/
│   └── Heart_Disease_Risk_Prediction_Deep_Learning_Probabilistic_Calibration.pdf
├── src/
│   ├── model.py           # Bagged Multi-Layer Perceptron architecture
│   └── calibration.py     # Platt Scaling & Isotonic Regression calibration engine
├── main.py                # Reliability evaluation & Brier score calculation demo
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start
```bash
git clone https://github.com/VIBHAS777/calibrated-heart-disease-risk-prediction.git
cd calibrated-heart-disease-risk-prediction
pip install -r requirements.txt
python main.py
```

---

## 📜 Citation
```bibtex
@article{venkatesh2026heartdisease,
  title={Heart Disease Risk Prediction Using Deep Learning and Probabilistic Calibration Techniques},
  author={Venkatesh, Bajjareddy and M R, Rashmi and K, Meera and Lee, Hoong Pine and Sagar B, Hima and Reddy, Araveeti Vibhas and Irigela, Bharghav Vardhan Reddy},
  journal={Engineered Science},
  volume={Online First},
  pages={ES-2026-0082},
  year={2026}
}
```
