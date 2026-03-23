# 🐛 Software Fault Prediction using Machine Learning

> A Django-based web application that predicts software defects in source code modules using multiple machine learning algorithms — helping teams identify fault-prone code before deployment.

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Django](https://img.shields.io/badge/django-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Dataset](#dataset)
- [ML Models Used](#ml-models-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## About

This project is a full-stack **Django web application** that applies machine learning to predict whether a software module contains defects. Users can register, log in, view the dataset, run ML model comparisons, and predict defects on custom input by entering software metrics manually through a web form.

The system uses the **NASA CM1 dataset** and trains multiple classifiers to evaluate defect-proneness based on Halstead and code complexity metrics.

---

## Features

- ✅ User Registration & Login with account activation workflow
- ✅ View dataset (first 100 rows of CM1)
- ✅ Automated ML pipeline: preprocessing → training → evaluation
- ✅ Four ML models compared: Naive Bayes, Random Forest, SVM, PCA+SVM
- ✅ 10-Fold Cross-Validation for each model
- ✅ Custom defect prediction via input form (18 software metrics)
- ✅ Complexity evaluation (Successful / Redesign classification)
- ✅ Interactive charts using Plotly (histogram, scatter, box plots)
- ✅ Correlation heatmap using Seaborn

---

## Dataset

The application uses the **NASA CM1** dataset (`cm1.csv`), placed in the `media/` directory.

| Dataset | Description | Source |
|---------|-------------|--------|
| CM1 | NASA spacecraft instrument software | NASA MDP / PROMISE Repository |

**Input Features used for prediction:**

| Feature | Description |
|---------|-------------|
| `loc` | Lines of Code |
| `n` | Program length (Halstead) |
| `v` | Volume (Halstead) |
| `l` | Program level (Halstead) |
| `d` | Difficulty (Halstead) |
| `i` | Intelligence content (Halstead) |
| `e` | Effort (Halstead) |
| `b` | Estimated bugs (Halstead) |
| `t` | Time to implement (Halstead) |
| `lOCode` | Lines of code only |
| `lOComment` | Lines of comments |
| `lOBlank` | Blank lines |
| `locCodeAndComment` | Lines with code and comments |
| `uniq_Op` | Unique operators |
| `uniq_Opnd` | Unique operands |
| `total_Op` | Total operators |
| `total_Opnd` | Total operands |
| `branchCount` | Branch count |

**Target:** `defects` → `true` (defective) or `false` (non-defective)

> Download CM1 dataset from the [PROMISE Repository](http://promise.site.uottawa.ca/SERepository/) and place it in the `media/` folder.

---

## ML Models Used

| Model | Key Parameters |
|-------|---------------|
| Naive Bayes | GaussianNB |
| Random Forest | `n_estimators=100`, `random_state=42` |
| SVM | `kernel='linear'`, `C=1` |
| PCA + SVM | `n_components=10`, StandardScaler + SVM linear |

All models use **10-Fold Cross-Validation** and are evaluated on Accuracy, Precision, and Recall.

---

## Project Structure

```
Software_Fault_Prediction/
│
├── Software_Fault_Predction/       # Django project settings
│   └── settings.py
│
├── app/                            # Main Django app
│   ├── models.py                   # UserRegistrationModel
│   ├── views.py                    # All views (auth, dataset, ML, prediction)
│   ├── forms.py                    # UserRegistrationForm
│   └── urls.py
│
├── templates/
│   ├── UserRegistrations.html
│   ├── UserLogin.html
│   └── users/
│       ├── UserHomePage.html
│       ├── viewdataset.html
│       ├── ml.html                 # ML results page
│       └── predictForm.html        # Custom prediction form
│
├── media/
│   └── cm1.csv                     # NASA CM1 dataset (place here)
│
├── db.sqlite3                      # SQLite database
├── manage.py
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/software-fault-prediction.git
cd software-fault-prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Apply database migrations
python manage.py migrate

# Place the CM1 dataset in the media folder
cp cm1.csv media/

# Run the development server
python manage.py runserver
```

**Key dependencies:**
```
django
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
chart_studio
imbalanced-learn
```

---

## Usage

### 1. Register & Login
- Visit `http://127.0.0.1:8000/register` to create an account
- An admin must activate your account before login is allowed
- Login at `http://127.0.0.1:8000/login`

### 2. View Dataset
- Navigate to **View Dataset** to browse the first 100 rows of the CM1 dataset

### 3. Run ML Models
- Navigate to **ML Analysis** to train and compare all four models
- Results displayed: Cross-validation scores, Accuracy, Precision, Recall

### 4. Predict Defects
- Navigate to **Predict Defect**
- Enter values for all 18 software metrics in the form
- The system trains a Random Forest model and predicts: **true** (defective) or **false** (non-defective)

---

## Results

Performance on the **CM1** dataset:

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Naive Bayes | 94% | 87% | 88.9% |
| Random Forest | 91% | 86% | 80.9% |
| SVM (Linear) | 94% | 85% | 91% |
| PCA + SVM | 94% | 85% | 91% |

> All models evaluated using 10-Fold Cross-Validation with an 80/20 train-test split.

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- [NASA Metrics Data Program (MDP)](https://nasa-softwaredefectdatasets.wikispaces.com/)
- [PROMISE Software Engineering Repository](http://promise.site.uottawa.ca/SERepository/)
- [scikit-learn](https://scikit-learn.org/) — ML models
- [Django](https://www.djangoproject.com/) — Web framework
- [Plotly](https://plotly.com/) — Interactive visualizations
