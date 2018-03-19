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

#This file shows type 2 baseline models. Data with the same latitude level in the Northern hemisphere
#and Southern hemisphere as testing data are training data. In this file, linear regression,
#Decision tree, random forest and gradient boosting regressor can be utilised for training and
#testing.

#load data
dataTraining = pd.read_csv('Dental_Traits_and_NPP.csv')

#Test data are Africa data
dataTesting = dataTraining.loc[(dataTraining['CONT']=='AF') & (dataTraining['lat_bio']<=12.74)]

dataTrain = dataTraining.loc[dataTraining['CONT']!='AF']
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
    if(i==2):
        data_realTrain = dataTrain.loc[(dataTrain['lat_bio'] > (5*i - 12.74)) & (dataTrain['lat_bio']<=(12.74 - 5*i))]
        print data_realTrain.lat_bio.count()
    if(i!=2):
        data_realTrain = dataTrain.loc[(dataTrain['lat_bio'] > (12.74 - 5*(i+1))) & (dataTrain['lat_bio']<=(12.74 - 5*i))]
        data_mirror =  dataTrain.loc[(dataTrain['lat_bio'] > (5*i - 12.74)) & (dataTrain['lat_bio'] <= (5*(i+1)-12.74))]
        data_realTrain = pd.concat([data_realTrain, data_mirror], ignore_index=True)
        print data_realTrain.lat_bio.count()
        
    data_realTest = dataTesting.loc[(dataTesting['lat_bio'] > (12.74 - 5*(i+1))) & (dataTesting['lat_bio']<=(12.74 - 5*i))]
    print data_realTest.lat_bio.count()
    numberTestData = data_realTest.lat_bio.count()
    numberTrainData = data_realTrain.lat_bio.count()
    ArrayTrain = np.append(ArrayTrain, numberTrainData)
    ArrayTest = np.append(ArrayTest, numberTestData)
    data_plot = pd.concat([data_realTrain, data_realTest], ignore_index=True)
    if(i==5):
        cm = plt.cm.get_cmap('gist_rainbow')
        lons = np.asarray(data_plot['lon_bio'])
        lats = np.asarray(data_plot['lat_bio'])#vmin=0, vmax=3000,
    
        #sc = plt.scatter(lons, lats,c=dataTraining['clustervalue'],linewidths=0, vmin=0, vmax=10, s=12, cmap=cm)
        sc = plt.scatter(lons, lats,c=data_plot['NPP'],linewidths=0, vmin=0, vmax=3000, s=12, cmap=cm)
        plt.colorbar(sc)
        #for i in range()
        map = Basemap(resolution='c')
        map.drawmapboundary()
        map.drawcountries(linewidth=1, 
                    linestyle='solid', 
                    color='black', 
                    zorder=30)
        map.drawcoastlines()
        plt.show()


    print i
    i = i + 1
    

    X_train = np.asarray(data_realTrain[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    y_train = np.asarray(data_realTrain['NPP'])
    X_test = np.asarray(data_realTest[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    y_test = np.asarray(data_realTest['NPP'])

    

    
    

    

    
    #linear regression
    #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=data_realTrain).fit()
    #predicty = est.predict(data_realTest[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    #DT
    #regr_1 = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
    #predicty = regr_1.predict(X_test)
    #Random Forest
    #clf = RandomForestRegressor(n_estimators=13).fit(X_train, y_train)
    #predicty = clf.predict(X_test)
    #GBR
    est = GradientBoostingRegressor(n_estimators=22, learning_rate=0.13, max_depth=4, random_state=0, loss='ls').fit(X_train, y_train)
    predicty = est.predict(X_test)

    errorRMSE = math.sqrt(np.sum((y_test - predicty)**2)/(numberTestData * 1.0))
    ArrayRMSE = np.append(ArrayRMSE, errorRMSE)
    errorMAE = np.sum(np.fabs(y_test - predicty))/(numberTestData * 1.0)
    ArrayMAE = np.append(ArrayMAE, errorMAE)

    
    predicty = np.asarray(predicty)
    prediction = np.concatenate((prediction, predicty))

print len(prediction)
real_ = np.asarray(dataTesting['NPP'])

#Measure performance of models
r2_compare = r2_score(real_, prediction)
numberofdata = dataTesting.CONT.count()
print numberofdata
print r2_compare
RMSE_ = math.sqrt(np.sum((real_ - prediction)**2)/(numberofdata * 1.0))
print RMSE_
MAE = np.sum(np.fabs(real_-prediction))/(numberofdata * 1.0)
print MAE
   


ind = np.arange(10)
avg_bar1 = ArrayTest
avg_bar2 = ArrayTrain


rects1 = plt.bar(ind, avg_bar1, 0.15, color='#ff0000',label='Test')
rects2 = plt.bar(ind + 0.15, avg_bar2, 0.15, color='#00ff00', label='Train')

high_point_x = []   
for i in range(0,10):
    single_bar_group={rects1[i].get_height():rects1[i].get_x() + rects1[i].get_width()/2.0,
                      rects2[i].get_height():rects2[i].get_x() + rects2[i].get_width()/2.0}

    height_list = list(single_bar_group.keys())
    height_list.sort(reverse=True)
    for single_height in height_list:
        high_point_x.append(single_bar_group[single_height])
        break

trend_line = plt.plot(high_point_x,ArrayRMSE,marker='o', color='#5b74a8', label='RMSE')
trend_line2 = plt.plot(high_point_x,ArrayMAE,marker='o', color='black', label='MAE')
plt.xlabel('layers')
plt.xticks(ind+0.15, ('layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'layer 6', 'layer 7', 'layer 8', 'layer 9', 'layer 10'))
plt.legend()
plt.show()
