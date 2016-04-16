###########
### The following file generates bags of instances for the 20Newsgroups data set (see description in paper)
##########
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class_pair_list = [("sci.space","sci.med"),("comp.sys.mac.hardware","comp.sys.ibm.pc.hardware"),
                   ("rec.sport.hockey",'rec.sport.baseball')]

### Select binary classification task (category pair)  
c1,c2 = class_pair_list[2]

#load train,test data, represent with TF-IDF
newsgroups_valid = fetch_20newsgroups(subset = "test",categories=[c1, c2])
y_valid = newsgroups_valid.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words="english")   
                                 
newsgroups_train = fetch_20newsgroups(subset = "train",categories=[c1, c2])
dd = list(newsgroups_train.data)
y_train = list(newsgroups_train.target)
y_train =np.array(y_train)
dd = np.array(dd)
                         
X_train = vectorizer.fit_transform(dd)
X_valid = vectorizer.transform(newsgroups_valid.data)

y_train = np.where(y_train==0,-1,1)  
y_valid = np.where(y_valid==0,-1,1)  


### CREATE BAGS
### bag_list -- dictionary mapping row indices to bags
### list_sizes_props -- dictionary giving for each bag its size and label proportion

    
positives = list(np.where(y_train==1)[0])
negatives = list(np.where(y_train==-1)[0])

p1 = 0.5
p2 = 0.3
p3 = 0.2
b_size1 = 200; b_size2 = 50; b_size3 = 100

list_sizes_props = {1:{b_size1:p1},2:{b_size1:p1},3:{b_size2:p2},4:{b_size2:p2},5:{b_size3:p3},6:{b_size3:p3}}

bag_list = {}
for list_ind,b in list_sizes_props.iteritems():
    size,prop = b.items()[0]
    n_pos = int(np.round(size*prop))
    n_neg = int(np.round(size*(1-prop)))
    labels_bag = [1]*n_pos + [-1]*n_neg #np.random.binomial(1,prop,size=size)
    bag_indices = []
    for l in labels_bag:
        if l == 1:
            ind = positives.pop()
            bag_indices.append(ind)
        elif l == -1:
            ind = negatives.pop()
            bag_indices.append(ind)
        else:
            print "label err"
            break;
    bag_list[list_ind] = bag_indices
