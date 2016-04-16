###########
### The following file generates bags of instances for the Movie data set (see description in paper)
##########

from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.cross_validation import train_test_split


#### PATH TO MOVIE DATA SET
### polarity dataset v2.0 : https://www.cs.cornell.edu/people/pabo/movie-review-data/

path_to_review_data_folder = ""

pos = path_to_review_data_folder +"\\txt_sentoken\\pos"
posfiles = [f for f in listdir(pos) if isfile(join(pos, f))]

neg = path_to_review_data_folder + "\\txt_sentoken\\neg"
negfiles = [f for f in listdir(neg) if isfile(join(neg, f))]

pos_reviews = []
for p in posfiles:
    with open(pos +"\\"+p,'rb') as infile:
        pos_review = infile.read()
        pos_reviews.append(pos_review)

neg_reviews = []
for n in negfiles:
    with open(neg +"\\"+n,'rb') as infile:
        neg_review = infile.read()    
        neg_reviews.append(neg_review)
len(pos_reviews)
len(neg_reviews)

##use 1 for positive sentiment, -1 for negative
y = np.concatenate((np.ones(len(pos_reviews)), -1*np.ones(len(neg_reviews))))
all_data  = np.concatenate((pos_reviews, neg_reviews))
indices = range(len(all_data))
shuf = np.random.shuffle(indices)
all_data = all_data[indices]
y = y[indices]


### train/test split
data_train, data_valid, y_train, y_valid = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y)


######## CREATE BAGS
### bag_list -- dictionary mapping row indices to bags
### list_sizes_props -- dictionary giving for each bag its size and label proportion
bag_list = {}
    
Wp = "great" 
Wm = "good"
Wb = "bad"
bag_list = {}
list_sizes_props = {}   
d0 = [i for i in range(len(data_train))  if (Wp in data_train[i].lower())]
bag_list[0] = d0
list_sizes_props[0] ={len(d0):float(sum(y_train[d0]))/(2*len(d0)) + 0.5} 
  
d1 = [i for i in range(len(data_train))  if (Wm in data_train[i].lower() and i not in d0)]
bag_list[1] = d1
list_sizes_props[1] ={len(d1):float(sum(y_train[d1]))/(2*len(d1)) + 0.5} 
  
d2 = [i for i in range(len(data_train))  if (Wb in data_train[i].lower() and i not in d0+d1)]
bag_list[2] = d2
list_sizes_props[2] ={len(d2):float(sum(y_train[d2]))/(2*len(d2)) + 0.5} 
  

for k,b in bag_list.iteritems():
    print k,len(b)
