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

#This file is for Type 1 Baseline models, data in the same level of test data are selected as training data

#Load data
dataTraining = pd.read_csv('Dental_Traits_and_NPP.csv')


#Data in Africa with latitude lager than 12.74 are not included in the testing data
dataTesting = dataTraining.loc[(dataTraining['CONT']=='AF') & (dataTraining['lat_bio']<=12.74)]

dataTrain = dataTraining.loc[(dataTraining['CONT']!='AF') & (dataTraining['lat_bio']<=12.74)]
min_lat_bio = dataTesting['lat_bio'].min()
dataTesting = dataTesting.sort_values(['lat_bio'],ascending = False)


print min_lat_bio
i = 0
prediction = np.array([])
ArrayRMSE = np.array([])
ArrayMAE = np.array([])
ArrayTrain = np.array([])
ArrayTest = np.array([])

while((12.74 - 5*i) >=  min_lat_bio):
    data_realTrain = dataTrain.loc[(dataTrain['lat_bio'] > (12.74 - 5*(i+1))) & (dataTrain['lat_bio']<=(12.74 - 5*i))]
    print data_realTrain.lat_bio.count()
    data_realTest = dataTesting.loc[(dataTesting['lat_bio'] > (12.74 - 5*(i+1))) & (dataTesting['lat_bio']<=(12.74 - 5*i))]
    print data_realTest.lat_bio.count()
    numberTestData = data_realTest.lat_bio.count()
    numberTrainData = data_realTrain.lat_bio.count()
    ArrayTrain = np.append(ArrayTrain, numberTrainData)
    ArrayTest = np.append(ArrayTest, numberTestData)

    i = i + 1
    print i

    X_train = np.asarray(data_realTrain[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    y_train = np.asarray(data_realTrain['NPP'])
    X_test = np.asarray(data_realTest[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    y_test = np.asarray(data_realTest['NPP'])

    
    
    
    
    ######linear regression#############################
    #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=data_realTrain).fit()
    #predicty = est.predict(data_realTest[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    ####################################################
    

    #Decision Tree
    #regr_1 = DecisionTreeRegressor(max_depth=25).fit(X_train, y_train)
    #predicty = regr_1.predict(X_test)
    
    #Random Forest
    #clf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
    #predicty = clf.predict(X_test)
    #GBR 0.13
    est = GradientBoostingRegressor(n_estimators=17, learning_rate=0.13, max_depth=2, random_state=0, loss='ls').fit(X_train, y_train)
    predicty = est.predict(X_test)

    errorRMSE = math.sqrt(np.sum((y_test - predicty)**2)/(numberTestData * 1.0))
    ArrayRMSE = np.append(ArrayRMSE, errorRMSE)
    errorMAE = np.sum(np.fabs(y_test - predicty))/(numberTestData * 1.0)
    ArrayMAE = np.append(ArrayMAE, errorMAE)
    
    
    predicty = np.asarray(predicty)
    prediction = np.concatenate((prediction, predicty))

print len(prediction)
real_ = np.asarray(dataTesting['NPP'])
r2_compare = r2_score(real_, prediction)
numberofdata = dataTesting.CONT.count()
print numberofdata
print r2_compare
RMSE_ = math.sqrt(np.sum((real_ - prediction)**2)/(numberofdata * 1.0))
print RMSE_
MAE = np.sum(np.fabs(real_-prediction))/(numberofdata * 1.0)
print MAE
   
#It prints error value over different layers
x = np.asarray([1,2,3,4,5,6,7,8,9,10])

plt.plot(x, ArrayRMSE, 'o', markersize=7, color='blue', alpha=0.5, label='RMSE')
plt.plot(x, ArrayRMSE)
plt.plot(x, ArrayMAE, 'o', markersize=7, color='red', alpha=0.5, label='MAE')
plt.plot(x, ArrayMAE)
plt.xlabel('layers')
plt.ylabel('error')
plt.title('error over different layers')
plt.legend()
plt.show()












