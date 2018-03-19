import pandas as pd
import numpy as np
import math
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from mpl_toolkits.basemap import Basemap, cm
#This file shows how to cluster the data



#load dental traits data
dataTraining = pd.read_csv('Dental_Traits_and_NPP.csv')

X = np.asarray(dataTraining[['mean_HYP','mean_LOP', 'mean_FCT_HOD', 'mean_FCT_AL', 'mean_FCT_OL', 'mean_FCT_SF', 'mean_FCT_OT', 'mean_FCT_CM']])

Z = linkage(X, 'ward')
c = pd.DataFrame(Z, columns=list('xyzn'))
c.to_csv("Z.csv",sep=',',encoding='utf-8')

k=10
clusteLabels = fcluster(Z, k, criterion='maxclust')
dataTraining['clustervalue'] = clusteLabels
dataTraining.to_csv("withlabels.csv",sep=',',encoding='utf-8')

#plot the dendrogram
i=0
NROW = dataTraining.CONT.count()
def llf(id):
    flag = 0
    if id < NROW:
        return str(id)
    else:
        
        return str(id)

plt.figure(figsize=(40, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('cluster index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=10,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    #leaf_label_func=llf,
    #leaf_rotation=90.,
    #leaf_font_size=12.,
    #show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

#plot clustering result on the map

cm = plt.cm.get_cmap('gist_rainbow')
lons = np.asarray(dataTraining['lon_bio'])
lats = np.asarray(dataTraining['lat_bio'])#vmin=0, vmax=3000,
    
#sc = plt.scatter(lons, lats,c=dataTraining['clustervalue'],linewidths=0, vmin=0, vmax=10, s=12, cmap=cm)
sc = plt.scatter(lons, lats,c=dataTraining['clustervalue'],linewidths=0, vmin=1, vmax=10, s=12, cmap=cm)
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
 

