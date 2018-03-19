import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
#from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#This file shows how to make predictions on cluster 1 of Africa data
#linear regression, decision trees, random forest and gradient boosting regressor 
#can be utilised

#Load the data
dataTraining = pd.read_csv('withlabels.csv')
#select the cluster 1 in Africa as testing data
dataTesting = dataTraining.loc[(dataTraining['CONT']=='AF')&(dataTraining['clustervalue']==1)]
numberTest = dataTesting.CONT.count()
print numberTest
#[1,2,3,4,7,8,9,10,5,6]
#Choose what clusters are included in the training data
clusterNumber = np.asarray([1,2,3,4,7,8,9,10,5,6])
total_Train = pd.DataFrame()
MAE_ = np.asarray([])
RMSE_ = np.asarray([])
numberofTraining = np.asarray([])
for i in range(len(clusterNumber)):
    dataTrain = dataTraining.loc[(dataTraining['CONT']!='AF')&(dataTraining['clustervalue']==clusterNumber[i])]
    print dataTrain.clustervalue.count()
    total_Train = pd.concat([total_Train, dataTrain], ignore_index=True)
    print total_Train.clustervalue.count()
    numberofTraining = np.append(numberofTraining, total_Train.clustervalue.count())
    
    X_train = np.asarray(total_Train[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    y_train = np.asarray(total_Train['NPP'])
    X_test = np.asarray(dataTesting[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    
    
    #linear regression
    #est = smf.ols(formula='NPP ~ mean_HYP + mean_LOP + mean_FCT_HOD+ mean_FCT_AL + mean_FCT_OL+ mean_FCT_SF+mean_FCT_OT+mean_FCT_CM', data=total_Train).fit()    
    #predicty = est.predict(dataTesting[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT','mean_FCT_CM' ]])
    

    
    #Decision Tree
    #regr_1 = DecisionTreeRegressor(max_depth=21).fit(X_train, y_train)
    #predicty = regr_1.predict(X_test)
    
    #Random Forest
    #clf = RandomForestRegressor(n_estimators=15).fit(X_train, y_train)
    #predicty = clf.predict(X_test)
    
    
    
    #Gradient Bossting Regressor
    est = GradientBoostingRegressor(n_estimators=3, learning_rate=0.0001, max_depth=3, random_state=0, loss='ls').fit(X_train, y_train)
    predicty = est.predict(X_test)
    
    
    real = np.asarray(dataTesting['NPP'])
    rmse = math.sqrt(np.sum((real-predicty)**2)/(numberTest*1.0))
    RMSE_ = np.append(RMSE_, rmse)
    #mae = np.sum(np.fabs(real-predicty))/(numberTest*1.0)
    #MAE_ = np.append(MAE_, mae)





print RMSE_
print MAE_


print len(predicty)
dataTesting['Predict_GBR'] = predicty
#Calculate RMSE and MAE
real = np.asarray(dataTesting['NPP'])
rmseF = math.sqrt(np.sum((real-predicty)**2)/(numberTest*1.0))
print rmseF
mae = np.sum(np.fabs(real-predicty))/(numberTest*1.0)
print mae






























##############plot the result on the world map###############################################################
'''
cm = plt.cm.get_cmap('gist_rainbow')
lons = np.asarray(dataAF['lon_bio'])
lats = np.asarray(dataAF['lat_bio'])#vmin=0, vmax=3000,
    
#sc = plt.scatter(lons, lats,c=dataTraining['clustervalue'],linewidths=0, vmin=0, vmax=10, s=12, cmap=cm)
sc = plt.scatter(lons, lats,c=dataAF['clustervalue'],linewidths=0, vmin=1, vmax=10, s=12, cmap=cm)
plt.colorbar(sc)
#for i in range()
map = Basemap(resolution='c')
map.drawmapboundary()
map.drawcountries(linewidth=1, 
            linestyle='solid', 
            color='white', 
            zorder=30)
map.drawcoastlines()
plt.show()
'''
###############################################################################################################
    
#####################code for Clustering data and draw dendrogram ###################################################################
'''
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
X = np.asarray(dataTraining[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT', 'mean_FCT_CM']])
#try_x = X[:151]
Z = linkage(X, 'ward')
c = pd.DataFrame(Z, columns=list('xyzn'))
c.to_csv("Z.csv",sep=',',encoding='utf-8')
#c, coph_dists = cophenet(Z, pdist(X))
#print c
k=10
clusteLabels = fcluster(Z, 10, criterion='maxclust')
dataTraining['clustervalue'] = clusteLabels
dataTraining.to_csv("withlabels.csv",sep=',',encoding='utf-8')
'''

'''
def llf(id):
    if id < 28886:
        return str(id)
    else:
        return str(id)

plt.figure(figsize=(40, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=10,  # show only the last p merged clusters
    #show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_label_func=llf,
    leaf_rotation=90.,
    leaf_font_size=12.,
    #show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()
'''
#######################################################################################################################################
