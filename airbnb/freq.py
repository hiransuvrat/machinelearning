__author__ = 'suvrat'

from csv import DictReader

csvFile = 'sessT.csv'
actionType = []
fieldName = 'action'
idField = 'user_id'
totalTime = []

for e, row in enumerate( DictReader(open(csvFile)) ):
    if not(row['action'] in actionType):
        #print row['action']
        actionType.append(row[fieldName])
    if (e%100000)==0:
        print e

print actionType

id = 'd1mm9tcy42'
outFile = 'timeTable.csv'
freq = [1.0] * len(actionType)
time = [0.0] * len(actionType)

with open(outFile,"wb") as outfile:
    outfile.write("%s,%s\n" % (idField,','.join(actionType)))
    for e, row in enumerate( DictReader(open(csvFile)) ):
        if row[idField] == id:
            freq[actionType.index(row[fieldName])] += 1
            if row['secs_elapsed'] is None or row['secs_elapsed'] == '':
                time[actionType.index(row[fieldName])] += 0
            else:
                time[actionType.index(row[fieldName])] += float(row['secs_elapsed'])
        else:
            outfile.write("%s,%s\n" % (id,','.join([str(i/j) for i,j in zip(time, freq)])))
            freq = [1.0] * len(actionType)
            time = [0.0] * len(actionType)
            id = row[idField]




