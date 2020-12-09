import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage import io
image=io.imread('sam.png')
io.imshow(image)
io.show()
print(image.shape)
rows=image.shape[0]
cols=image.shape[1]
image=image.reshape(image.shape[0]*image.shape[1],4)
print(image.shape)
kmeans=KMeans(n_clusters=64,n_init=1,max_iter=200,verbose=1)
kmeans.fit(image)
print(kmeans.cluster_centers_)
clusters=np.asarray(kmeans.cluster_centers_,dtype=np.uint8)
print(clusters.shape)
labels=np.asarray(kmeans.labels_,dtype=np.uint8)
print(labels.shape)
print(labels)
o=pd.DataFrame(labels,columns=['cen'])
o.to_csv('labels.csv')
labels=labels.reshape(rows,cols)
np.save('codebook.sam.npy',clusters)
io.imsave('compressed_sam.png',labels)
#CONSTRUCTION OF COMPRESSED IMAGE
newimage=np.zeros((rows,cols,4), dtype= np.uint8)
for i in range(rows):
    for j in range(cols):
        newimage[i, j, :]=clusters[labels[i, j], :]
io.imsave('sam compress.png', newimage)

