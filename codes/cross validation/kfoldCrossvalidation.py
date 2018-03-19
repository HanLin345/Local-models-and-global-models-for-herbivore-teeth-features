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

#python version for normal 11 folds cross validation
#for linear regression, Decision trees, random forest, gradient boosting regressor
#Parameters for regressors are:
#Decision Tree: max_depth it can be any integer value > 0
#Random Forest: n_estimators it can be any integer value > 0
#Gradient Tree Boosting:
#n_estimators: integer > 0, learning_rate: real number > 0, max_depth: integer > 0
 

if __name__ == '__main__':
    
    data = pd.read_csv('Dental_Traits_and_NPP.csv')
    print data.head(15)
    
    shuffleData = data.sample(frac = 1).reset_index(drop = True)
    print shuffleData.head(15)
    K = 11
    numberDataTest = 28886/11
    prediction = np.array([])
    for i in range(K):
        test_fold = shuffleData[(numberDataTest*i):(numberDataTest*(i+1))]
        
        
        if(i == 0):
            
            train_1 = shuffleData[(numberDataTest*(i+1)):]
            
            X_train = np.asarray(train_1[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(train_1['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            ######Then training and testing can be here#######
            #OLS
            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=train_1).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            #Decision Tree
            #regr_1 = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
            #predicty = regr_1.predict(X_test)
            #Random Forest
            #clf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
            #predicty = clf.predict(X_test)
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)
            
            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            #########Don't forget to store the prediction value in an array#########
        if(i == 10):
            
            train_1 = shuffleData[:(numberDataTest*i)]
            X_train = np.asarray(train_1[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(train_1['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            #OLS
            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=train_1).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            #Decision Tree
            #regr_1 = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
            #predicty = regr_1.predict(X_test)
            #Random Forest
            #clf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
            #predicty = clf.predict(X_test)
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))

        if(i>0 and i<10):
            train_left = shuffleData[:(numberDataTest*i)]
            train_right = shuffleData[(numberDataTest*(i+1)):]
            trainTotal = pd.concat([train_left, train_right], ignore_index=True)

            X_train = np.asarray(trainTotal[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(trainTotal['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            #OLS
            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=trainTotal).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            #Decision Tree
            #regr_1 = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
            #predicty = regr_1.predict(X_test)
            #Random Forest
            #clf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
            #predicty = clf.predict(X_test)
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            
    
    print len(prediction)
    shuffleData['Predict_GradientT'] = prediction
    
    
    realPrecipitation = np.asarray(shuffleData['NPP'])
    r2_compare = r2_score(realPrecipitation, prediction)
    print r2_compare
    RMSE_ = math.sqrt(np.sum((realPrecipitation-prediction)**2)/(28886.0))
    print RMSE_
    MAE = np.sum(np.fabs(realPrecipitation-prediction))/(28886.0)
    print MAE
 