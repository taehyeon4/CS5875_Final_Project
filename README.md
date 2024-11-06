# Applied Machine Learning (CS5875) - Final Project

Mid-way Write Up: [https://docs.google.com/document/d/1dhnugiaBpyexRTNT6Ghatg2bwr8zQ5svmRu-R_yIhbk/edit?usp=sharing](https://docs.google.com/document/d/1dnPzyLG72VDvTBUiUiyWv12h3RLcgTJV-Uqc2XPTmUI/edit?usp=sharing)

Final Write Up: https://docs.google.com/document/d/1dhnugiaBpyexRTNT6Ghatg2bwr8zQ5svmRu-R_yIhbk/edit?usp=sharing

Dataset: https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data

The dataset is very large (4.5 gb) and hence for prototyping our code, we made a *MiniData* set where we sampled 10 random images from each letter in order to train on local. Based on the reuslts of this, we tweaked our code and understood our convergence values (F1, Confusion Matrix). 


Workflow: 


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
 - if we have enough bandwith we can work on sentence correction, fixing typos, and proper grammar
