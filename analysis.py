import scipy.stats as stats
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import math 
import scipy as sp

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

#Plot graphs
def plotData(x, y, alpha=0.4, s=20):
    gt = plt.plot([i for i in range(len(y))], y, alpha=alpha, label='ground truth')
    plt.scatter([i for i in range(len(y))], x, s=s, alpha=alpha)
    plt.xlim((0, len(y)))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

#Evaluate the regression
def computeCost(x, y, w, constant=0, costType = 'log'):
    #Number of samples
    samples = y.size
    if costType == 'log':
        predReg = x.dot(w).flatten()
        predVal = []
        for itrPred in predReg:
            #Calculate predicted values
            predVal.extend([(1.0/(1+math.pow(math.e, (-1*(constant+itrPred)))))])
        #Calculate log loss
        error = logLoss(y, predVal)
        return error

#Calculate log loss
def logLoss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

#Compute losses across grid of weights
def genGridLoss(x, y, w1Range=[0,1], breaks = 100):
    #Grid over which we will calculate J
    theta0_vals = np.linspace(w1Range[0], w1Range[1], breaks)

    #initialize J_vals to a matrix of 0's
    JVals = np.zeros(shape=(theta0_vals.size))
 
    #Fill out JVals
    for t1, element1 in enumerate(theta0_vals):
        thetaT = np.zeros(shape=(1, 1))
        thetaT[0][0] = element1
        JVals[t1] = computeCost(x, y, [0, element1])

    return JVals, theta0_vals

def genCountourPlot(x, y, weightRange = [0, 1], breaks = 10):
    JVals, w = genGridLoss(x, y, weightRange, breaks)
    #Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    #contour(w, JVals, logspace(-2, 3, 20))
    plt.xlabel('weight')
    plt.ylabel('loss')
    plt.plot(w, JVals)
    plt.show()

#genCountourPlot(np.array([[1],[1], [1]]),np.array([0, 1, 0]), [-1,0], 1000)
