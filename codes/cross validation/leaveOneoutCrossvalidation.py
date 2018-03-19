from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from numpy import meshgrid
import statsmodels.formula.api as smf
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import math

#This file is for leave one out cross validation for linear regression,
#Decision tree, random forest and gradient boosting regressor

if __name__ == '__main__':
    #Read data
    data = pd.read_csv('Dental_Traits_and_NPP.csv')
    
    shuffleData = data.sample(frac = 1).reset_index(drop = True)
    #K is the total number of data
    K = 28886
    numberDataTest = 1
    prediction = np.array([])
    for i in range(K):
        test_fold = shuffleData[(numberDataTest*i):(numberDataTest*(i+1))]
        
        if(i == 0):
            
            train_1 = shuffleData[(numberDataTest*(i+1)):]

            X_train = np.asarray(train_1[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(train_1['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            
            ######Then training and testing can be here#######
            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=train_1).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)
            
            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            #########Don't forget to store the prediction value in an array#########
        if(i == 28885):
            
            train_1 = shuffleData[:(numberDataTest*i)]

            X_train = np.asarray(train_1[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(train_1['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=train_1).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)
            
            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))

        if(i>0 and i<28885):
            train_left = shuffleData[:(numberDataTest*i)]
            train_right = shuffleData[(numberDataTest*(i+1)):]
            trainTotal = pd.concat([train_left, train_right], ignore_index=True)

            X_train = np.asarray(trainTotal[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(trainTotal['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=trainTotal).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            
    print prediction
    print len(prediction)
    real = np.asarray(shuffleData['NPP'])
	#Measure performance of models on all test folds by using R square, RMSE and MAE
    r2_compare = r2_score(real, prediction)
    print r2_compare
    RMSE_ = math.sqrt(np.sum((real-prediction)**2)/(28886.0))
    print RMSE_
    MAE = np.sum(np.fabs(real-prediction))/(28886.0)
    print MAE
    

