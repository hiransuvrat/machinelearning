__author__ = 'suvrat'

import pandas as pd

resultFile = './air/sampleRes.csv'
h2oFile = './air/h2ogbm.csv'

h2o = pd.read_csv(h2oFile)
result = []
prob = []
res = pd.read_csv(resultFile)
res['prob'] = 1

for itr in range(0, len(h2o.index)):
    tempResult = {}
    for col in h2o.columns:
        if col != 'predict':
            tempResult[col] = h2o[col][itr]
    sortedCountry = sorted(tempResult, key=tempResult.__getitem__, reverse=True)
    sortedValue = sorted(tempResult.values(), reverse=True)
    result.extend(sortedCountry[:7])
    prob.extend(sortedValue[:7])
print len(res.index), len(result)
res['country'] = result
res['prob'] = prob
#res = res.drop('prob', axis=1)
res.to_csv('./air/gbm.csv', index=False)
