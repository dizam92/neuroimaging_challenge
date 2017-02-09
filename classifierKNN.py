import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("./datasets/train.csv")
test = pd.read_csv("./datasets/test.csv")


data['Diagnosis'] = data['Diagnosis'].replace({'MCI':'r', 'cMCI':'b', 'AD':'g', 'HC':'k'})
test = test.drop(['SUB_ID', 'GENDER'], axis=1)

# datas[0] contient les outliers
# datas[1] ne contient plus les 3 rangees outliers
datas = (data, data.drop([17, 152, 203], axis=0))

i = 0

for dataset in datas:
    data_diagnostic = dataset['Diagnosis']
    data = dataset.drop(['SUB_ID','Diagnosis','GENDER'], axis=1)


    pcaobj = PCA(n_components=3)
    new_data = pcaobj.fit_transform(data)
    #print np.sum(pcaobj.explained_variance_ratio_)
    plt.figure()
    plt.scatter(new_data[:,0], new_data[:,1], c=data_diagnostic)
    plt.xlabel('Comp1')
    plt.ylabel('Comp2')
    plt.show()

new_test = pcaobj.transform(test)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(new_test[:,0], new_test[:,1], zs=new_test[:,2])
ax.set_xlim([-0.15e7,0.15e7])
ax.set_ylim([-0.15e7,0.15e7])
ax.set_zlim([-0.15e7,0.15e7])

plt.show()

np.savetxt("newdata.csv", X=new_data, delimiter=",")
