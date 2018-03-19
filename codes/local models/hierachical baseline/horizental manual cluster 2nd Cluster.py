import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
#from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#This file shows how to cut the cluster 2 into five horizontal layers 
#And test different layers separately.
#linear regression, decision tree, random forest and gradient boosting regressor can be tested

#Load data
dataTraining = pd.read_csv('withlabels.csv')
#Select cluster 1 in Africa as testing data
dataTesting = dataTraining.loc[(dataTraining['CONT']=='AF') & (dataTraining['clustervalue']==2)]
dataTesting = dataTesting.loc[(dataTesting['lat_bio']<=14.99)&(dataTesting['lat_bio']>=-14.54)]

min_lat_bio = dataTesting['lat_bio'].min()
dataTesting = dataTesting.sort_values(['lat_bio'],ascending = False)

#Cut cluster 2 into 5 layers
print min_lat_bio
i = 5#the maximum value of i is 5
#prediction = np.array([])
#while((12.74 - 5*i) >=  min_lat_bio):

#Select a layer as real test data for making prediction
data_realTest = dataTesting.loc[(dataTesting['lat_bio'] > (14.99 - 5*(i+1))) & (dataTesting['lat_bio']<=(14.99 - 5*i))]
numberTest = data_realTest.lat_bio.count()
print numberTest



#[2,1,3,4,7,8,9,10,5,6]
#Select a group of data that are not in Africa as training data
clusterNumber = np.asarray([2])
total_Train = pd.DataFrame()
RMSE_ = np.asarray([])
MAE_ = np.asarray([])
numberofTraining = np.asarray([])
for j in range(len(clusterNumber)):
    dataTrain = dataTraining.loc[(dataTraining['CONT']!='AF')&(dataTraining['clustervalue']==clusterNumber[j])]
    print dataTrain.clustervalue.count()
    total_Train = pd.concat([total_Train, dataTrain], ignore_index=True)
    print total_Train.clustervalue.count()
    numberofTraining = np.append(numberofTraining, total_Train.clustervalue.count())
    
    X_train = np.asarray(total_Train[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    y_train = np.asarray(total_Train['NPP'])
    X_test = np.asarray(data_realTest[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    


    
    
    #regr_1 = DecisionTreeRegressor(max_depth=10).fit(X_train, y_train)
    #predicty = regr_1.predict(X_test)
    
    
    #clf = RandomForestRegressor(n_estimators=15).fit(X_train, y_train)
    #predicty = clf.predict(X_test)
    
    
    
    
    est = GradientBoostingRegressor(n_estimators=12, learning_rate=0.02, max_depth=4, random_state=0, loss='ls').fit(X_train, y_train)
    predicty = est.predict(X_test)
    
    
    
    
    



    #linear regression
    
    #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=total_Train).fit()
    #predicty = est.predict(data_realTest[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    
    
    real = np.asarray(data_realTest['NPP'])
    rmse = math.sqrt(np.sum((real-predicty)**2)/(numberTest*1.0))
    RMSE_ = np.append(RMSE_, rmse)
    mae = np.sum(np.fabs(real-predicty))/(numberTest*1.0)
    MAE_ = np.append(MAE_, mae)
    

print RMSE_

print len(predicty)
data_realTest['Predict'] = predicty

real = np.asarray(data_realTest['NPP'])
#Calculate RMSE and MAE for a layer
rmseF = math.sqrt(np.sum((real-predicty)**2)/(numberTest*1.0))
print rmseF
mae = np.sum(np.fabs(real-predicty))/(numberTest*1.0)
print mae
