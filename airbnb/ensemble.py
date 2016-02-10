import pandas as pd

data = []
#fileList = ['./air/xgb0.csv', './air/xgb1000.csv']#, './air/xgb5000.csv', './air/xgb50000.csv', './air/xgb1050000.csv']
fileList = ['./air/xgb.csv', './air/gbm.csv', './air/rf.csv']#, './etc/etc.csv']
noFile = len(fileList)
bestOf = 7
scoreType = 'ranking'

for fileName in fileList:
    data.append(pd.read_csv(fileName))

result = pd.read_csv('./air/finalSampleRes.csv')
print len(result.index)
scoreResult = []
count = 0
tempBest = []
score = []
rank = bestOf

for itrData in range(0, len(data[0].index)):
    rank -= 1 
    count += 1
    for itrFile in range(0, noFile):
        tempBest.append(data[itrFile]['country'][itrData])
        if scoreType == 'ranking':
            scoreData = rank
        else:
            scoreData = data[itrFile]['prob'][itrData]
        score.append(scoreData)
    if (count%bestOf) == 0:
        rank = bestOf
        finalScore = {}
        countScore = {}
        leastScore = 1
        for itrProb in range(0,len(tempBest)):
            if leastScore > score[itrProb]:
                leastScore = score[itrProb]
            if tempBest[itrProb] in finalScore:
                finalScore[tempBest[itrProb]] += score[itrProb]
                countScore[tempBest[itrProb]] += 1
            else:
                finalScore[tempBest[itrProb]] = score[itrProb]
                countScore[tempBest[itrProb]] = 1
        for key, value in countScore.iteritems():
            if value <= bestOf:
                finalScore[key] += ((bestOf - value)*leastScore)
        sortedCountry = sorted(finalScore, key=finalScore.__getitem__, reverse=True)
        #print (itrData-bestOf+1),itrData,sortedCountry,finalScore
        #print result['country'][(itrData-bestOf+1):(itrData+1)]
        scoreResult.extend(sortedCountry[:5])
        tempBest = [] 
        score = []
        
print len(scoreResult), len(result.index)
result['country'] = scoreResult
#result = result.drop('prob', axis=1)
result.to_csv('./air/ensemble.csv', index=False)
            
