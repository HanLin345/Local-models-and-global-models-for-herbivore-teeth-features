from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from numpy import meshgrid
import statsmodels.formula.api as smf
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import math
#This file is for spatial leave one out cross validation for linear regression, decision tree and 
#random forest, gradient boosting regressor


def calculate_delta_lat(D, R):
    delta_lat = math.degrees(D/R)
    return delta_lat

def calculate_delta_lon(D, R, max_lat):
    delta_lon = math.degrees(D/(R*(math.cos(math.radians(max_lat)))))
    return delta_lon
	
#This function is for calculating geographical distance between two data points on earth
def calculateDistance(R,newLat,newlon, testlat, testlon):
    mean_lat = math.radians((newLat + testlat)/2.0)
    delta_lat = math.radians(newLat-testlat)
    delta_lon = math.radians(newlon-testlon)
    distance = R*(math.sqrt(delta_lat**2 + (math.cos(mean_lat)*delta_lon)**2))
    return distance

#This function is for discarding data that are near the test point
def spacial_crossvd(R, D, train_1, threshold_lat_above, threshold_lat_below, threshold_lon_left, threshold_lon_right):
    
    trainingR1above = train_1.loc[(train_1['lat_bio']>threshold_lat_above) | (train_1['lat_bio']<threshold_lat_below)]
    trainingR2restleft = train_1.loc[(train_1['lat_bio']<=threshold_lat_above) & (train_1['lat_bio']>=threshold_lat_below) & (train_1['lon_bio']<threshold_lon_left)]
    trainingR2restright = train_1.loc[(train_1['lat_bio']<=threshold_lat_above) & (train_1['lat_bio']>=threshold_lat_below) & (train_1['lon_bio']>threshold_lon_right)]
    trainingReal = pd.concat([trainingR1above, trainingR2restleft], ignore_index=True)
    trainingReal = pd.concat([trainingReal, trainingR2restright], ignore_index=True)
    
    trainingFilter = train_1.loc[(train_1['lat_bio']<=threshold_lat_above) & (train_1['lat_bio']>=threshold_lat_below) & (train_1['lon_bio']>=threshold_lon_left) & (train_1['lon_bio']<=threshold_lon_right)]
    newRownumber = trainingFilter.lon_bio.count()
    
    trainingFilter.index = range(newRownumber)
    k = 0
    for k in range(newRownumber):
        distances = calculateDistance(R,trainingFilter.at[k,'lat_bio'],trainingFilter.at[k,'lon_bio'], test_fold['lat_bio'], test_fold['lat_bio'])
        #print distances
        if distances < D:
            trainingReal = trainingReal.append(trainingFilter.loc[k], ignore_index=True)
        else:
            continue
    print trainingReal.lon_bio.count()
    return trainingReal


if __name__ == '__main__':
    #Load data
    data = pd.read_csv('Dental_Traits_and_NPP.csv')
    #K is 28886 means it is leave one out cross validation
    K = 28886
    numberDataTest = 1
	#D: Distance between test data point and the nearest data points that are not be dropped.
    D = 500.0
	#R is the radius of the earth
    R = 6371.0
    delta_lat = calculate_delta_lat(D, R)
    
    prediction = np.array([])
    for i in range(K):
        print "loop ", i
        test_fold = data[(numberDataTest*i):(numberDataTest*(i+1))]
        
        threshold_lat_above = np.asscalar(test_fold['lat_bio'] + delta_lat)
        threshold_lat_below = np.asscalar(test_fold['lat_bio'] - delta_lat)
        if math.fabs(threshold_lat_above) > math.fabs(threshold_lat_below):
            max_lat = threshold_lat_above
        else:
            max_lat = threshold_lat_below
        
        delta_lon = calculate_delta_lon(D, R, max_lat)
        
        threshold_lon_left = np.asscalar(test_fold['lon_bio'] - delta_lon)
        threshold_lon_right = np.asscalar(test_fold['lon_bio'] + delta_lon)
        
        if(i == 0):
            
            train_1 = data[(numberDataTest*(i+1)):]

            trainingReal = spacial_crossvd(R, D, train_1, threshold_lat_above, threshold_lat_below, threshold_lon_left, threshold_lon_right)
            
            X_train = np.asarray(trainingReal[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(trainingReal['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    



            
            ######Then training and testing can be here#######
            #est = smf.ols(formula='bio12 ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=trainingReal).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            #########Don't forget to store the prediction value in an array#########
        
        if(i == 28885):
            
            train_1 = data[:(numberDataTest*i)]

            trainingReal = spacial_crossvd(R, D, train_1, threshold_lat_above, threshold_lat_below, threshold_lon_left, threshold_lon_right)

            X_train = np.asarray(trainingReal[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(trainingReal['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            
            ######Then training and testing can be here#######
            #est = smf.ols(formula='bio12 ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=trainingReal).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)

            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            #########Don't forget to store the prediction value in an array#########
        
        if(i>0 and i<28885):
            train_left = data[:(numberDataTest*i)]
            train_right = data[(numberDataTest*(i+1)):]
            train_1 = pd.concat([train_left, train_right], ignore_index=True)
            trainingReal = spacial_crossvd(R, D, train_1, threshold_lat_above, threshold_lat_below, threshold_lon_left, threshold_lon_right)
            
            X_train = np.asarray(trainingReal[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
            y_train = np.asarray(trainingReal['NPP'])
            X_test = np.asarray(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            
            ######Then training and testing can be here#######
            #est = smf.ols(formula='bio12 ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=trainingReal).fit()
            #predicty = est.predict(test_fold[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])

            #Gradient Tree Boosting
            est = GradientBoostingRegressor(n_estimators=7, learning_rate=1.2, max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
            predicty = est.predict(X_test)
            
            predicty = np.asarray(predicty)
            prediction = np.concatenate((prediction, predicty))
            #########Don't forget to store the prediction value in an array#########
           
    print prediction
    print len(prediction)
    real = np.asarray(data['NPP'])
	
	#Following are indices for measuring performance of a model
    r2_compare = r2_score(real, prediction)
    print r2_compare
    RMSE_ = math.sqrt(np.sum((real-prediction)**2)/(28886.0))
    print RMSE_
    MAE = np.sum(np.fabs(real-prediction))/(28886.0)
    print MAE
