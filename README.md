# EEG Sleep Stage Classification
This project presents an end-to-end implementation of sleep stage classification from PSG/EEG signals using two different approaches: EEGNet (deep learning) and YASA (feature-based machine learning), evaluated on a subset of the Sleep Physionet dataset. The dataset consists of full-night PSG recordings from healthy subjects, segmented into 30-second epochs and classified into five sleep stages (W, N1, N2, N3, REM).

## Dataset
This corresponds to a subset of 153 recordings from 37 males and 41 females that were 25-101 years old at the time of the recordings.

[MNE Datasets](https://mne.tools/stable/documentation/datasets.html#the-sleep-polysomnographic-database)
[Fetch Function](https://mne.tools/stable/generated/mne.datasets.sleep_physionet.age.fetch_data.html#mne.datasets.sleep_physionet.age.fetch_data)
[Example Implementation](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html#tut-sleep-stage-classif)

## Installation Guide

### A. Notebook
Presents the results of an initial exploratory data analysis (EDA) of the polysomnography (PSG) dataset used in the Machine Learning course assignment. The analysis focuses on understanding the dataset structure, sleep stage label distribution, raw signal quality, and EEG signal frequency characteristics prior to the preprocessing and deep learning modeling stages.

In addition, this notebook also covers the complete pipeline including data preprocessing, model building, training, and evaluation of two different approaches for sleep stage classification: EEGNet and YASA.

#### Local

##### 1. Create Virtual Environment (Optional, but recommended)
```
python -m venv venv
venv\Scripts\activate
```

##### 2. Install dependencies
```
pip install numpy scipy pandas scikit-learn matplotlib seaborn tensorflow lightgbm
```

##### 3. Run Notebook.ipynb

#### Google Colab

##### 1. Run Notebook.ipynb

### B. Apps
The application accept input (e.g., sample EEG data or simulated data) and display the prediction/classification results produced by the model.

#### 1. Install dependencies
```
pip install -r Apps/requirements.txt
```

#### 2. Streamlit run app.py