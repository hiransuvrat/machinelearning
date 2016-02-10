__author__ = 'suvrat'

from csv import DictReader
import pandas as pd

def avg(inputFile):
    data = pd.read_csv(inputFile)
    val = []
    for col in data.columns:
        
        if col != 'user_id':
            val.append(data[col].mean)
            print data[col].mean()
    return val

def genFile(inputFile, outFile):
    with open(outFile,"wb") as outfile:
        for e, row in enumerate(DictReader(open(inputFile))):
            #Write headers to the file
            if e == 0:
                outfile.write("id")
                for header in row:
                    if header != 'id':
                        outfile.write(",%s" % (header))
                for sessFeat in range(0,sessFeatures):
                    outfile.write(",t" + str(sessFeat))
                outfile.write("\n")

            outfile.write("%s" % (row['id']))
            for header in row:
                if header != 'id':
                    outfile.write(",%s" % (row[header]))
            if row['id'] in id:
                outfile.write(",%s\n" % (','.join([str(i) for i in data[id[row['id']]]])))
            else:
                outfile.write("%s\n" % ('').join([str(i) for i in zeros]))


csvFile = './air/freqTable.csv'
sessFeatures = 360
trainOutFile = './air/sessTrain.csv'
testOutFile = './air/sessTest.csv'
trainFile = './air/train_users_2.csv'
testFile = './air/test_users.csv'
#trainFile = './air/joinedTrain7.csv'
#testFile = './air/joinedTest7.csv'
data = []
id = {}
zeros = [','] * sessFeatures

#val = avg(csvFile)

for e, row in enumerate( DictReader(open(csvFile)) ):
    data.append([])
    id[row['user_id']] = e
    for header in row:
        if header != 'user_id':
            data[e].append(row[header])

genFile(trainFile, trainOutFile)
print 'train generated'
genFile(testFile, testOutFile)

