from __future__ import division
import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import numpy as np
import sklearn.tree as tree
import sklearn.naive_bayes as bayes
import sklearn.metrics as metrices
import math

#Calculates weighted gini score
def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight}) 
    df = df.sort('pred',ascending=False) 
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight 
    #return df.gini.sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)

#Test the data to predict for positive and negative sample
def classification(data, featuresList):
    data['target'][:, (data['target'] > 0)] = 1
    data['target'][:, (data['target'] == 0)] = 0

    data, testa, features, fillVal = util.prepDataTrain(data, 'target', featuresList, True, 50, True, True, 'median', False, 'set')
    print 'Data preped'

    clf = bayes.GaussianNB()
    #clf = tree.DecisionTreeClassifier()
    clf.fit(data[features].tolist(), data['target'])
    pred = clf.predict_proba(testa[features].tolist())[:, 1]
    pred[pred > .005] = 1
    pred[pred <= .005] = 0
    res = testa['target'] - pred
    print res, pred, testa['target'], len(np.where(res[res < -.5])[0]), len(np.where(res[res > .5])[0]), len(np.where(testa['target'][testa['target'] > .5])[0]), testa.shape, data.shape
    #scores = cross_validation.cross_val_score(clf, data[features].tolist(), data['target'], cv=5, scoring='recall')
    #print scores

data = np.genfromtxt('../ImpVarSmoothTrain.csv', names=True, delimiter=',')
X_test = np.genfromtxt('../ImpVarSmoothtest.csv', names=True, delimiter=',')
data1 = np.copy(data)

featuresList= ['weatherVar185','weatherVar21','weatherVar189','weatherVar161','weatherVar103','weatherVar95','weatherVar194','weatherVar216','weatherVar186','weatherVar110','weatherVar137','weatherVar23','weatherVar49','weatherVar232','weatherVar68','weatherVar22','weatherVar151','weatherVar16','geodemVar14','geodemVar29','var8','var4','var10','var11','var12','var13','var15','var17']

#Cross validation testscores
for i in ([0, 1]):
    data = np.copy(data1)
    data, testa, features, fillVal = util.prepDataTrain(data, 'target', featuresList, True, 50, False, True, 'median', False, 'set', i)
    data['target'] = np.log(math.e + data['target'])
    data['target'][data['target'] > 3] = 3 #np.log(data['target'][data['target'] > 10])

    print 'Data preped'

    clf = ensemble.GradientBoostingRegressor(n_estimators=45, max_depth=5, min_samples_leaf=20, min_samples_split=30, verbose=True, loss='ls')
    clf.fit(data[features].tolist(), data['target'])
    print 'fitted'


    pred = np.power(clf.predict(testa[features].tolist()), math.e)
    print normalized_weighted_gini(testa['target'],pred,testa['var11'])
    #for i in range(len(clf.feature_importances_)):
    #    print i, clf.feature_importances_[i], features[i]


#Carry out building data on full model
data = np.copy(data1)
data, testa, features, fillVal = util.prepDataTrain(data, 'target', featuresList, False, 50, False, True, 'median', False, 'set', i)
data['target'] = np.log(math.e + data['target'])

#Final predictions
clf.fit(data[features].tolist(), data['target'])
test = util.prepDataTest(X_test, features, True, fillVal, False, 'set')
pred = np.power(clf.predict(test[features].tolist()), math.e) #clf.predict(test[features].tolist())
df = pd.DataFrame({"id": X_test['id'], "target": pred})
df.to_csv("predictions.csv", index=False, cols=["id", "target"])
