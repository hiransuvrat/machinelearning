from sklearn import decomposition
import numpy as np
import pandas as pd

pd.set_option('precision', 3)

data = pd.read_csv("/home/air/timeTable.csv")
userId = data['user_id']
data = data.drop(['user_id'], axis=1)

pca = decomposition.PCA(n_components=7)
data = pca.fit_transform(data)

print pca.explained_variance_ratio_

data = pd.DataFrame(data)
data['user_id'] = userId
data.to_csv("pcaTime7.csv", index=False)

