# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:37:41 2017

@author: Neltherion

"""
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot 
#read data
dataframe=pd.read_fwf('brain_body.txt')
print (dataframe)

#choose features and labels
feature=dataframe[['Brain']]
label=dataframe[['Body']]

#train model 
predictor=linear_model.LinearRegression()
predictor.fit(feature,label)

print(predictor.predict(feature))

#show graphs aka data visualization
matplotlib.pyplot.scatter(feature,label)
matplotlib.pyplot.plot(feature,predictor.predict(feature))
matplotlib.pyplot.show