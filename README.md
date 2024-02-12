# The Day After Tomorrow - WindOracle tool

## :hammer_and_wrench: Tool Introduction

This is a tool desinged for predicting the satellite images of tropical storm​ and speed wind. The main goal of this prediction tool is to output the next images of a series of time series storms and the corresponding wind speeds of the images. In this tool, we predict tropical storm images through **seq2seq** with each a CNN-LSTM in both encoder and decoder, and predict the wind speed of tropical storm through CNN.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)](https://www.python.org/downloads/release/python-3110/)


## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-vincent/edit/main/requirements.txt)

## :warning: This project uses conda as version control tool

other used libraries
- [Numpy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [scikit-learn](https://scikit-learn.org)
- [matplotlib](https://matplotlib.org)
- [pytorch]([/#](https://pytorch.org/))
- [pytest](https://docs.pytest.org/en/8.0.x/)
- [flake8](https://flake8.pycqa.org/en/latest/)
- prtorch_msssim
- tqdm
- livelossplot


## 🗃️: Module Introduction

The tool seperated into main sections, predicting images and wind speed. We used package `Storm` to do the whole process.

- **Predicting images - victor_functions**
  - data_loader.py
  - datasets.py
  - model.py `making models`
  - training.py `training the model`
  - image_loader.py
  - image_model.py
- **Predicting wind speed - StormForcast**
  - datasets.py `making dataset and dataloader`
  - models.py `making models`
  - train.py `training the model`
  - utilities.py
  
``` 
Storm/
│
├── victor_functions/  
│   ├── __init__.py
│   ├── data_loader.py
│   ├── datasets.py
│   ├── image_loader.py
│   ├── image_model.py
│   ├── model.py
│   └── training.py
│
└── StormForcast/     
    ├── __init__.py
    ├── datasets.py
    ├── models.py
    ├── train.py
    └── utilities.py
```
## 🚀&nbsp; Software Installation Guide

1. Clone the repo
```
$ git clone https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-vincent
```

2. Change your directory to the cloned repo 
```
$ cd acds-the-day-after-tomorrow-vincent
```

3. Create a Python virtual environment named 'vincent' and activate it
```
$ conda create -n vincent
```
```
$ conda activate vincent
```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip install -r requirements.txt
```

or, you can directly run the following code instead, it will run the <code>setup.py</code> automatically to collect the resources in this file and install mcsim for you

```
$ pip install -e .
```

## 📖 User Guide
**package - StormForcast:**

```python
from Storm.StormForcast import Train_Validate

# Load 10 storms, without resume, for original dataset, not for surprise storm:
cnn_model = StormForcast.Train_Validate(
        '/content/gdrive/MyDrive/gp2/Selected_Storms_curated/',
        task='WindSpeed', device='cuda',
        batch_size_train=32, batch_size_val=1000, batch_size_test=1000,
        lr=2e-3, epoch=100, split_method='random', num_storms=10)
                              
# Load 1 storm, with resume, not for surprise storm
resume_model = StormForcast.Train_Validate(
        '/content/gdrive/MyDrive/gp2/Selected_Storms_curated/',
        task='WindSpeed', device='cuda',
        batch_size_train=32, batch_size_val=1000, batch_size_test=1000,
        lr=2e-3, epoch=100, split_method='random',
        num_storms=1, resume=True, resume_path='CNNGeneral_epoch_16.pth')

# Load surprise storms, with resume:
surprise_model = StormForcast.Train_Validate(
        './tst/tst', task='WindSpeed', device='cuda',
        batch_size_train=32, batch_size_val=100, batch_size_test=100,
        lr=2e-3, epoch=20, split_method='random', num_storms=3, surprise_storm=True,
        resume=True, resume_path='CNNGeneral_epoch_16.pth')

# train the network (used for both retrain or train from scratch)
surprise_model.train_whole()

# this function evaluate the model according to the test set of dataset
surprise_model.evaluate()

# this function predict the unlabled images in surprise storm
surprise_model.predict()

# this function draw the result from predict or evaluate
surprise_model.draw_result(type='predict' / type='evaluate)
```
**Notebook Guide**

- EDA.ipynb `Try to do the explored data analysis first among 30 storms.`
- Pipeline.ipynb `Making predictions on **Surpirse storm**`
- predictions.csv `The result of wind speed on **Surpirse storm**`
- Task1.ipynb `Exploration and demonstration of task 1`
- Task2_wind_speed_prediction.ipynb `Exploration and demonstration of task 2`


## :blue_book: Documentation

The code includes a basic [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be built by running

```bash
python -m sphinx docs html
```

then viewing the generated `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual PDF can be generated by running

```bash
python -m sphinx  -b latex docs latex
```

Then follow the instructions to process the `FloodTool.tex` file in the `latex` directory in your browser.


## :hourglass: Testing

The tool includes several tests, which you can use to check its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```bash
python -m pytest --doctest-modules test
```


## 📅 Dataset 

**Link to the training dataset**
https://drive.google.com/drive/folders/1tFqoQl-sdK6qTY1vNWIoAdudEsG6Lirj?usp=drive_link

**Link to the briefing slides**
https://imperiallondon.sharepoint.com/:p:/r/sites/TrialTeam-EA/Shared%20Documents/General/TheDayAfterTomorrow-presentation%202.pptx?d=wdf1d9e0210264eab88858e2353a36242&csf=1&web=1&e=XoU1Am

## 🧾 Reference

https://drivendata.co/blog/predict-wind-speeds-benchmark/

https://www.drivendata.org/competitions/72/predict-wind-speeds/page/275/

https://drivendata.co/blog/predict-wind-speeds-benchmark/


## :writing_hand: Owner

Vincent Team Member:
- Yu, Qinhan
- Leng, Peiyi
- Hu, Zhuojun
- Li, Rui
- Shi, Sizhe
- Petala, Naya
- Odusi, Barin
- Ondrusz, Chris R
