# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:13:18 2022

@author: Tuf
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import missingno as msno
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Sequential 
from tensorflow.keras.layers import BatchNormalization, Dense,Dropout 
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.utils import plot_model   
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from modules_for_customer_segmentation import EDA,model_evaluation

#%% STATICS
CSV_PATH = os.path.join(os.getcwd(),'Dataset','Train.csv')
JOB_ENCODER_PATH =os.path.join(os.getcwd(),'Model', 'Job_Encoder.pkl')
MARITAL_ENCODER_PATH =os.path.join(os.getcwd(),'Model', 'Marital_Encoder.pkl')
EDUCATION_ENCODER_PATH =os.path.join(os.getcwd(),'Model',
                                     'Education_Encoder.pkl')
DEFAULT_ENCODER_PATH =os.path.join(os.getcwd(),'Model', 'Default_Encoder.pkl')
HOUSING_ENCODER_PATH =os.path.join(os.getcwd(),'Model','Housing_Encoder.pkl')
PERSONAL_ENCODER_PATH =os.path.join(os.getcwd(),'Model',
                                    'Personal_Encoder.pkl')
COMMUNICATION_ENCODER_PATH =os.path.join(os.getcwd(),'Model',
                                         'Communication_Encoder.pkl')
MONTH_ENCODER_PATH =os.path.join(os.getcwd(),'Model', 'Month_Encoder.pkl')
PREVCAMPAIGN_ENCODER_PATH =os.path.join(os.getcwd(),'Model',
                                        'Prevcampaign_Encoder.pkl')
OHE_PATH =os.path.join(os.getcwd(),'Model','ohe.pkl')
SS_FILE_NAME =os.path.join(os.getcwd(),'Model','standard_scaler.pkl')
LOG_PATH=os.path.join(os.getcwd(),'Model','Logs',datetime.datetime.now().\
                      strftime("%Y%m%d-%H%M%S"))

#%% Functions
def cramers_corrected_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% EDA
#Step 1: Data loading
df=pd.read_csv(CSV_PATH)

#%%
#Step 2: Data Inspection
#1) descriptive statistics
stats=df.describe().T
df.boxplot()

#2) check for outliers, NaNs and duplicates
df.duplicated().sum()
df.isna().sum() 

#3) Clean unnecessary column
df=df.drop(['id', 'days_since_prev_campaign_contact'], axis=1)
df.info() 

#4) Visualize the distribution of dataset
df.info()
#cat_column=df.columns[df.dtypes=='object']
cat_column=['job_type', 'marital', 'education', 'default', 'housing_loan',
       'personal_loan', 'communication_type', 'month','prev_campaign_outcome']
con_column=['customer_age','balance','day_of_month','last_contact_duration',
            'num_contacts_in_campaign','num_contacts_prev_campaign']

eda=EDA()
eda.plot_cat(df,cat_column)  
eda.plot_con(df,con_column) 

#Based on the initial inspection:
# The data consists of NaNs in column:
# customer_age, marital ,balance,personal_loan, last_contact_duration, 
# num_contacts_in_campaign, days_since_prev_campaign_contact

# Based on msno.bar plot, days_since_prev_campaign_contact has extreme sum of
# missing number. Therefore, the id and days_since_prev_campaign_contact are 
# dropped from this dataset
# Based on the plot:
# the marital status of married has the highest number of no term deposit subscribed
# compared than other status which was probably due to housing and personal loan.
# As for the education, the secondary education also showed no term deposit
# subscribed as probably duelower income as shown in job type.
    
#%% Step 3: Data Cleaning

#1) Label Encoder for categorical data

le=LabelEncoder()  
      
paths=[JOB_ENCODER_PATH,MARITAL_ENCODER_PATH,EDUCATION_ENCODER_PATH,
       DEFAULT_ENCODER_PATH,HOUSING_ENCODER_PATH,PERSONAL_ENCODER_PATH,
       COMMUNICATION_ENCODER_PATH,MONTH_ENCODER_PATH,PREVCAMPAIGN_ENCODER_PATH]

for index,i in enumerate(cat_column):
    temp=df[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df[i]=pd.to_numeric(temp,errors='coerce')
    with open(paths[index],'wb') as file:
        pickle.dump(le,file)
        

#3) Imputation of NaNs
df.isna().sum()

# categorical data
df['marital']=df['marital'].fillna(df['marital'].mode()[0])
df['personal_loan']=df['personal_loan'].fillna(df['personal_loan'].mode()[0])
# continuous data
df['customer_age']=df['customer_age'].fillna(df['customer_age'].median())
df['balance']=df['balance'].fillna(df['balance'].median())
df['last_contact_duration']=df['last_contact_duration'].\
    fillna(df['last_contact_duration'].median())
df['num_contacts_in_campaign']=df['num_contacts_in_campaign'].\
    fillna(df['num_contacts_in_campaign'].median())

df.info()
df.isna().sum()
df.duplicated().sum()
#
#%% Step 4: Features selection

#Cramer's V categorical vs categorical
for i in cat_column:
    print(i)
    confussion_mat= pd.crosstab(df[i],df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))

#Continuous vs categorical
from sklearn.linear_model import LogisticRegression
for con in con_column:    
        print(con)
        lr=LogisticRegression()
        lr.fit(np.expand_dims(df[con],axis=-1),df['term_deposit_subscribed'])
        print(lr.score(np.expand_dims(df[con],axis=-1),
                       df['term_deposit_subscribed']))


#From feature selections, customer_age(0.89),balance(0.89),day_of_month(0.89),
# last_contact_duration(0.89), num_contacts_in_campaign(0.89) and 
# num_contacts_prev_campaign(0.89) are selected for model training.
#%%
#Step 5: Data pre-processing
X=df.loc[:,['customer_age','balance','day_of_month','last_contact_duration',
                  'num_contacts_in_campaign', 'num_contacts_prev_campaign']]
y=df['term_deposit_subscribed']

#1)  Target dataset(OneHotEncoder)
ohe = OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=-1))
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file) 
# Eventhough the target variable is already in 0 and 1, OHE is conducted
# to avoid the target to have relationship with no(continuous no)

#2) Features scalling
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X= std.fit_transform(X)
with open(SS_FILE_NAME,'wb') as file:
    pickle.dump(std,file)
    
#3) test split test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                  random_state=123)

#%% Model Development 
nb_classes=len(np.unique(y_train,axis=0)) #2
nb_features = np.shape(X_train)[1:] #6

#1) SEQUENTIAL API
model=Sequential()
model.add(Input(shape=(nb_features),name='InputLayer')) 
model.add(Dense(128,activation='relu',name='HiddenLayer1'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu',name='HiddenLayer2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(nb_classes,activation='softmax',name='OutputLayer'))
model.summary()

plot_model(model,show_shapes=True,show_layer_names=(True))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics='acc') 

# callbacks
#1) tensorboard callback
TensorBoard_callback = TensorBoard(log_dir=LOG_PATH)

#2) earlystopping
early_stopping_callback = EarlyStopping(monitor='loss',patience=3)

#%% Model training
hist = model.fit(x=X_train,y=y_train,
                  batch_size=64,
                  epochs=100,
                  validation_data=(X_test,y_test),
                  callbacks=[TensorBoard_callback, early_stopping_callback])

#%% Model Evaluation
hist.history.keys()

me=model_evaluation()
me.plot_graph(hist)

#From the graph, it shows that the model looks
#%% model Analysis
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import ConfusionMatrixDisplay

y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(X_test),axis=1)

cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true,y_pred)

print(cr)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
#%% model saving
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'Model','model.h5')
model.save(MODEL_SAVE_PATH)
#%% Discussion

# The accuracy of this model in predicting the term_deposit_subscribed was  
# trained against the selected features. 
# In the model development, despite additional of no of nodes and dense layer,
# the model improved insignificantly.
# Despite that, the model is believed to have a good ability in predicting 
# the term_deposit#_subscribed price as indicated by higher f1 score of 90%
# Hence, the model is a great predictive model which was probably due to
# a good dataset.
