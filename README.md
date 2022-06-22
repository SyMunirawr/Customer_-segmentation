# Customer-Segmentation
 About 32,000 data was trained for predictive model development.
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)


# Project Title
This project is to predict the outcome of the campaign. 

## Description
In this project, deep learning approach was used to train the dataset in predicting the  the outcome of the marketing campaign if the customers would like to term the deposit. 
After the training has completed, the model was evaluated and return a result of an accuracy value of 90%.

## Step-by-step
The dataset contained about 13,200 number of samples which consisted of 18 variables. The exploratory data analysis was conducted to filter out the best parameters which are important in predicting the likeliness the outcome of the marketing campaign if the customers would like to term the deposit. 
In the feature selections, logistic regression and Cramer's V analysis were conducted. Several parameters were chosen as they have higher correlations value (>0.8) against the target variable.
After filtering out the variable, the pre-processing step consisted of model development where 128 number of nodes, 2 dense layers with dropout and batchnormalized layer were added to train the model as in the model architecture. Despite the addition number of nodes and hidden layers, the model insignificantly improve above 90% as observed in the Tensorboard .

![model_architecture](https://user-images.githubusercontent.com/107612253/175026126-08ac406e-9b70-4e83-8ac3-8dcefea4805f.png)

![Tensorboard](https://user-images.githubusercontent.com/107612253/175025962-cb908b8d-9b4c-4581-9b63-376356f73d44.png)

However, the loss function was minimized with the additions. The loss andaccuracy of model was plotted in Tensorboard to visualize the trained model over a period of time. The model achieved 90% in accuracy which can be considered as a great predictive model which was probably due to a good dataset.

![f1 score](https://user-images.githubusercontent.com/107612253/175025783-8a29fb4d-8714-4291-a0b2-a314af1269ee.png)


## Acknowledgement
I would like to sincerely thank Mr.Kunal Gupta for the dataset which can be downloaded from [Kaggle][https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon).
