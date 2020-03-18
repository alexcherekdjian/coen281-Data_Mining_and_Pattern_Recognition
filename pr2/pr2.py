import csv
import random
import os
import numpy as np
import pandas as pd
from modlamp.ml import train_best_model
from modlamp.datasets import load_ACPvsRandom
from modlamp.descriptors import PeptideDescriptor
from modlamp.datasets import load_custom
from modlamp.ml import train_best_model, score_cv

# read in the dataset
df = pd.read_csv(
    filepath_or_buffer='test.dat', 
    header=None, 
    sep=',')

# separate names from classes
vals = df.iloc[:,:].values

samples_test = []
for i in range(0, len(vals)):
    samples_test.append(vals.item(i)[:])

# read in the dataset
_df = pd.read_csv(
    filepath_or_buffer='train.dat', 
    header=None, 
    sep=',')

# separate names from classes
vals = _df.iloc[:,:].values

y_train = []
samples_train = []
for i in range(0, len(vals)):
    y_train.append(vals.item(i)[0:2])
    samples_train.append(vals.item(i)[3:])

for i in range(0,len(y_train)):
    if y_train[i] == '1\t':
        y_train[i] = '1'
        
# reformatting training file for input into library data as csv
class_in = 0      
out = open('formatted.csv', 'w')
out.write('1566, 1, 1, -1\n')

i = 0
for x,y in zip(samples_train, y_train):
    # unable to read samples, skipping
    if i == 484 or i == 620:
        i+=1
        continue
    
    i+=1
    
    if y == '1':
        class_in = 1
    elif y == '-1':
        class_in = 0
    
    out.write(x + ', ' + str(class_in))
    out.write('\n')
out.close()


# load the reformatted data
data = load_custom(os.getcwd() +'/formatted.csv')

# create descriptors for peptide sequences
descr_temp = PeptideDescriptor(data.sequences, scalename='pepArc')
descr_temp.calculate_crosscorr(window=4)

# develop best model and print out score with cross validation
best_RF = train_best_model('RF', descr_temp.descriptor, data.target)
score_cv(best_RF, descr_temp.descriptor, data.target, cv=10)

y_pred = []

# get predictions for values
for i in range (0, 392):
    try:
        pep_descr = PeptideDescriptor(samples_test[i], scalename='pepArc')
        pep_descr.calculate_crosscorr(window=4)
        proba = best_RF.predict_proba(pep_descr.descriptor)
        y_pred.append(proba)

    except:
        print("Couldn't predict %d, assigning random prediction" %i)
        y_pred.append(random.choice(y_pred))

final_pred = []

# store final predictions from formatting done in csv
for val in y_pred:
    class_1prob = val[0][0]
    class_neg1prob = val[0][1]

    if class_1prob > class_neg1prob:
        final_pred.append(-1)
    else:
        final_pred.append(1)

# write predictions to a file
f = open("pred.txt", "w")
for i in range (0, len(final_pred)):
    f.write(str(final_pred[i]) + "\n")
f.close()

print("Writing predictions to file complete")