# Applied Machine Learning (CS5875) Final Project: Real-Time Sign Language Translator

## Authors
- **Akhil Reddy** (ar2537@cornell.edu)
- **Taehyeon Lim** (tl892@cornell.edu)

## Project Overview
This project currently implements a real-time system for translating sign language into text and speech, providing an inclusive communication tool for the hearing-impaired community. The system integrates sign language recognition, content moderation, and text-to-speech synthesis into one package to enable easy communication in video conferencing situations.

## Features
- **Real-Time Sign Language Recognition**: Detects and translates hand signs into text.
- **Grammatical Correction**: Automatically corrects punctuation, capitalization, and typos.
- **Content Moderation**: Filters inappropriate language and replaces it with contextually appropriate alternatives.
- **Text-to-Speech Conversion**: Converts the processed text into audible speech.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (python package manager)

## Installation and Setup

Follow these steps to download, install, and run the project.

### 1. Clone the Repository
Start by cloning the repository to your local machine in your desired directory:
```bash
git clone https://github.com/taehyeon4/CS5875_Final_Project.git
cd CS5875_Final_Project
```

### 2. Set Up a Virtual Environment
To isolate dependencies, please create a venv:
```bash
python -m venv venv
source venv/bin/activate (mac/linux)
venv\Scripts\activate (windows)
```

### 3. Install Dependencies
Install all required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the Project
To execute the pipeline, use the following command:
```bash
python main.py
```

### 4 (cont). Run the Project w/ pre-existing video
To execute the pipeline with a video (mp4), use the following command:
```bash
python main.py --input <path_to_input>
```

## Write-Ups
Please reference our midway and final write ups: 
- [Midway](https://docs.google.com/document/d/1dnPzyLG72VDvTBUiUiyWv12h3RLcgTJV-Uqc2XPTmUI/edit?usp=sharing)
- [Final](https://docs.google.com/document/d/1dnPzyLG72VDvTBUiUiyWv12h3RLcgTJV-Uqc2XPTmUI/edit?usp=sharing)

The original dataset that we used is from [Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data). It is very large (4.5 gb) and hence for prototyping our code, we made a *MiniData* set where we sampled a varying number of random images from each letter in order to train. Based on the reuslts of the current model, we decided how to scale up the model by understanding our convergence values (F1, Confusion Matrix). 

<!-- Workflow: 


Pytorch CNN: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/4e865243430a47a00d551ca0579a6f6c/cifar10_tutorial.ipynb

1) Download the entire dataset onto google drive
2) make a subset of the larger dataset for use to get smaller dataset for us to do training / initial testing on
3) host all of our code and training / testing data on google colab and train a large model
4) we can start testing with some of our own images, by passing in some more of images that we take ourselves
5) work on taking the images and converting it into proper words and a proper sentence
6) convert it into a live video model



for halfway point (nov. 11th) - 
1) motivation, method, premliminary experiments, future work - make a google document and write it up /  write up with everything we are planning to do
2) show training and loss for the smaller dataset and how it is going to scale up for when we train in google colab with GPU. talk about how GPU will help us get faster
3) do research on taking letters and spaces and converting that into words and sentences
4) what is left in our project
 - training the big dataset through google colab
 - functioning conversion from images to sentences
 - finally, we will do videos to sentences

Extra (if time permits): 
 - if we have enough bandwith we can work on sentence correction, fixing typos, and proper grammar -->
