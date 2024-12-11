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

### 3. Install Dependencies and download models

Install all required Python packages using `pip` and download the YOLO models:

```bash
pip install -r requirements.txt
```

```bash
cd vision/models
bash download-models.sh
```

### 4. Run the Project

To execute the pipeline, use the following command:

```bash
python main.py
```

Use -h flag to see all options.

<!-- ### 4 (cont). Run the Project w/ pre-existing video
To execute the pipeline with a video (mp4), use the following command:
```bash
python main.py --input <path_to_input>
``` -->

## Write-Ups

Please reference our midway and final write ups:

- [Midway](https://github.com/taehyeon4/CS5875_Final_Project/blob/main/writeups/Midway%20Write%20Up.pdf)
- [Final](https://github.com/taehyeon4/CS5875_Final_Project/blob/main/writeups/Final%20Write%20Up.pdf)

The original dataset that we used is from [Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data). It is very large (4.5 gb) and hence for prototyping our code, we made a _MiniData_ set where we sampled a varying number of random images from each letter in order to train. Based on the reuslts of the current model, we decided how to scale up the model by understanding our convergence values (F1, Confusion Matrix).

## Demo

https://github.com/user-attachments/assets/94c84b0a-7845-416d-9b9e-37490d57f76a
