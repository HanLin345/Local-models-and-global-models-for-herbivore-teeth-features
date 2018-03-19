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

#python version of spatial 11 folds cross validation
#for linear regression, Decision trees, random forest, gradient boosting regressor
#Parameters for regressors are:
#Decision Tree: max_depth it can be any integer value > 0
#Random Forest: n_estimators it can be any integer value > 0
#Gradient Tree Boosting:
#n_estimators: integer > 0, learning_rate: real number > 0, max_depth: integer > 0

#This function is for discarding data whose distance to the threshold is smaller than 300 kilometers
def spacialCVright(train_1, threshold):
    
    min_latRest = train_1['lat_bio'].min()
    max_latRest = train_1['lat_bio'].max()
    
    if math.fabs(max_latRest) > math.fabs(min_latRest):
        max_temp = max_latRest
    else:
        max_temp = min_latRest
                
    delta_long = 300.0/(111.0*math.cos(math.radians(max_temp)))
            
    
    if (train_1['lon_bio'].max() > (threshold + delta_long)):
        realTrain = train_1.loc[lambda df:df.lon_bio > (threshold + delta_long)]
        filterTrain = train_1.loc[lambda df:df.lon_bio <= (threshold + delta_long)]
        
        filterTrain.index = range(filterTrain.lon_bio.count())
        
    
        for i in range(filterTrain.lon_bio.count()):
            delta_y = 300.0/(111.0*math.cos(math.radians(filterTrain.loc[i,'lat_bio']))) + threshold
            if filterTrain.at[i,'lon_bio'] > delta_y:
                
           
                realTrain = realTrain.append(filterTrain.loc[i], ignore_index=True)
            else:
                continue
        
        return realTrain

    else:
        train_1.index = range(train_1.lon_bio.count())
        for i in range(train_1.lon_bio.count()):
            delta_y = 300.0/(111.0*math.cos(math.radians(train_1.loc[i,'lat_bio']))) + threshold
            if train_1.at[i,'lon_bio'] <= delta_y:
                
           
                train_1 = train_1.drop(i, inplace=True)
            else:
                continue
        
        return train_1
		
#This function is for discarding data whose distance to the threshold is smaller than 300 kilometers
def spacialCVleft(train_1, threshold):
    
    min_latRest = train_1['lat_bio'].min()
    max_latRest = train_1['lat_bio'].max()
    
    if math.fabs(max_latRest) > math.fabs(min_latRest):
        max_temp = max_latRest
    else:
        max_temp = min_latRest
                
    delta_long = 300.0/(111.0*math.cos(math.radians(max_temp)))
    
    if (train_1['lon_bio'].min() < (threshold - delta_long)):
        realTrain = train_1.loc[lambda df:df.lon_bio < (threshold - delta_long)]
        filterTrain = train_1.loc[lambda df:df.lon_bio >= (threshold - delta_long)]
        
        filterTrain.index = range(filterTrain.lon_bio.count())
        
        i = 0
        for i in range(filterTrain.lon_bio.count()):
            
            delta_y = threshold - 300.0/(111.0*math.cos(math.radians(filterTrain.loc[i,'lat_bio'])))
            if filterTrain.at[i,'lon_bio'] < delta_y:
                
           
                realTrain = realTrain.append(filterTrain.loc[i], ignore_index=True)
            else:
                continue
        
        return realTrain

    else:
        train_1.index = range(train_1.lon_bio.count())
        
        for i in range(train_1.lon_bio.count()):
            print i
            delta_y = threshold - 300.0/(111.0*math.cos(math.radians(train_1.loc[i,'lat_bio']))) 
            if train_1.at[i,'lon_bio'] >= delta_y:
                
           
                train_1 = train_1.drop(i, inplace=True)
            else:
                continue
        
        return train_1





if __name__ == '__main__':
    
	#Read data
    data = pd.read_csv('Dental_Traits_and_NPP.csv')
    
    minLongitude =  data['lon_bio'].min()
    data = data.sort_values(['lon_bio'])
    rowNumber = data.lon_bio.count()
    data.index = range(rowNumber)
    #spatial 11-fold cross validation
    K = 11
    numberDataTest = rowNumber/11
    prediction = np.array([])
    for i in range(K):
        test_fold = data[(numberDataTest*i):(numberDataTest*(i+1))]
        
        if(i == 0):
            
            train_1 = data[(numberDataTest*(i+1)):]
            threshold = test_fold.loc[(numberDataTest*(i+1))-1,'lon_bio']
            rightTrainData = spacialCVright(train_1, threshold)
            print rightTrainData.lon_bio.count()

            X_train = np.asarray(rightTrainData[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(rightTrainData['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            ######Then training and testing can be here#######
            #OLS
            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=rightTrainData).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            #Decision Tree
            #regr_1 = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
            #predicty = regr_1.predict(X_test)
            #Random Forest
            #clf = RandomForestRegressor(n_estimators=25).fit(X_train, y_train)
            #predicty = clf.predict(X_test)
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            #########Don't forget to store the prediction value in an array#########
        if(i == 10):
            
            train_1 = data[:(numberDataTest*i)]
            
            threshold = test_fold.loc[(numberDataTest*i),'lon_bio']
            
            leftTrainData = spacialCVleft(train_1, threshold)
            print leftTrainData.lon_bio.count()

            X_train = np.asarray(leftTrainData[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(leftTrainData['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            
            #OLS
            #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=leftTrainData).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            #Decision Tree
            #regr_1 = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
            #predicty = regr_1.predict(X_test)
            #Random Forest
            #clf = RandomForestRegressor(n_estimators=25).fit(X_train, y_train)
            #predicty = clf.predict(X_test)
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))

        if(i>0 and i<10):
            train_left = data[:(numberDataTest*i)]
            thresholdleft = test_fold.loc[(numberDataTest*i),'lon_bio']
            leftTrainData = spacialCVleft(train_left, thresholdleft)
            train_right = data[(numberDataTest*(i+1)):]
            thresholdright = test_fold.loc[(numberDataTest*(i+1))-1,'lon_bio']
            rightTrainData = spacialCVright(train_right, thresholdright)
            trainTotal = pd.concat([rightTrainData, leftTrainData], ignore_index=True)
            print trainTotal.lon_bio.count()

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
            #clf = RandomForestRegressor(n_estimators=25).fit(X_train, y_train)
            #predicty = clf.predict(X_test)
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))

    print prediction
    print len(prediction)
    
    
    realPrecipitation = np.asarray(data['NPP'])
    r2_compare = r2_score(realPrecipitation, prediction)
	
	#R square, Root mean squared error and mean absolute error for all test folds 
    print r2_compare
    RMSE_ = math.sqrt(np.sum((realPrecipitation-prediction)**2)/(28886.0))
    print RMSE_
    MAE = np.sum(np.fabs(realPrecipitation-prediction))/(28886.0)
    print MAE
    
 