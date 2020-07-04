'''
import csv

with open('/Users/Vibek/Desktop/UTP-Bydgoszcz/Anomaly_detection_packages/NSL-KDD/KDDTest+.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('/Users/Vibek/Desktop/UTP-Bydgoszcz/Anomaly_detection_packages/NSL-KDD/KDD_Test.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        #writer.writerow(('KDD_train', 'dataset'))
        writer.writerows(lines)

import pandas as pd
import os

df_chunked = pd.read_csv("myLarge.csv", chunksize=100)  # you can alter the chunksize

for chunk in df_chunked:
    uniques = chunk['col'].unique().tolist()
    for val in uniques:
        df_to_write = chunk[chunk['col'] == val]
        if os.path.isfile('small_{}.csv'.format(val)):  # check if file already exists
            df_to_write.to_csv('small_{}.csv'.format(val), mode='a', index=False, header=False)
        else:
            df_to_write.to_csv('small_{}.csv'.format(val), index=False)

'''
import csv

# your input file (10GB)
in_csvfile = open('/home/vibek/Anomanly_detection_packages/LITNET-2020/ALLinONE/allFlows.csv', "r")

# reader, that would read file for you line-by-line
reader = csv.DictReader(in_csvfile)

# number of current line read
num = 0

# number of output file
output_file_num = 1

# your output file
out_csvfile = open('out_{}.csv'.format(output_file_num), "w")

# writer should be constructed in a read loop, 
# because we need csv headers to be already available 
# to construct writer object
writer = None

for row in reader:
    num += 1

    # Here you have your data line in a row variable

    # If writer doesn't exists, create one
    if writer is None:
        writer = csv.DictWriter(
            out_csvfile, 
            fieldnames=row.keys(), 
            delimiter=",", quotechar='"', escapechar='"', 
            lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC
        )

    # Write a row into a writer (out_csvfile, remember?)
    writer.writerow(row)

    # If we got a 10000 rows read, save current out file
    # and create a new one
    if num > 2000000:
        output_file_num += 1
        out_csvfile.close()
        writer = None

        # create new file
        out_csvfile = open('out_{}.csv'.format(output_file_num), "w")

        # reset counter
        num = 0 





###Feature importnace####

model = ExtraTreesClassifier()
model.fit(XX_train,yy_train)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_importances = pd.Series(model.feature_importances_, index=indices)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

model.fit(train_data,test_data)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_importances = pd.Series(model.feature_importances_, index=indices)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

model.fit(train_data_i,test_data_i)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_importances = pd.Series(model.feature_importances_, index=indices)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

# Closing the files
in_csvfile.close()
out_csvfile.close()
