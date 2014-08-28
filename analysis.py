import scipy.stats as stats
from sklearn.decomposition import PCA
import numpy as np

#Calculate and print correlation values
def calcCorr(data, fields= []):
    if len(fields) == 0:
        fields = data.dtype.names
        
    completed = ['1'] * len(fields)
    count = -1
    for i in fields:
        count += 1
        for j in fields:
            if not (j in completed or i == j):
                print i, j, stats.pearsonr(data[i], data[j])
        completed[count] = i

#Find PCA object or get transformed values
def calcPCA(data, fields, nComponents= 3, transform = False):
    pca = PCA(n_components=nComponents)
    if transform:
        return pca.fit_transform(data[fields].tolist())
    else:
        return pca.fit(data[fields].tolist())

#Fill NaNs with mean, median or constant
def fillna(data, typeSub = 'mean', fields = [], constantVal = -1):
    if len(fields) == 0:
        fields = data.dtype.names
    
    fillVal = {}
    for field in fields:
        indices = np.where(np.isnan(data[field])) 
        if typeSub == 'mean':
            fillVal[field] = np.mean(data[field][~np.isnan(data[field])])
        elif typeSub == 'median':
            fillVal[field] = np.median(data[field][~np.isnan(data[field])])
        elif typeSub == 'constant':
            fillVal[field] = constantVal
        data[field][indices] = fillVal[field]
                    
    return data, fillVal

#Fill NaNs from dict
def fillnaDict(data, fields):
    for field in fields:
        indices = np.where(np.isnan(data[field])) 
        data[field][indices] = fields[field]

    return data


