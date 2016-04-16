###########
### The following file generates bags of instances for the Adult data set (see description in paper)
##########

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.cross_validation import KFold

def binarize_df(dframe):
    """ Binarize a pandas series of categorical strings into a sparse dataframe. """

    dfout = pd.DataFrame()
    for column in dframe.columns:
        col_dtype = dframe[column].dtype
        if col_dtype == object:
            # assume categorical string
            for category in dframe[column].value_counts().index:
                dfout[category] = (dframe[column] == category)
        elif col_dtype == np.int64  or col_dtype == np.float:
            dfout[column] = dframe[column]
        else:
            print "unused column: {}".format(column)
    return dfout
    
names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                 "hours-per-week", "native-country", "label"]

########## Set features and bags
features_to_keep = ["age","education", "race", "sex","hours-per-week","capital-gain","capital-loss", "label"]
bags = [' Masters',' Bachelors', ' Some-college', ' HS-grad']

#### PATH TO ADULT DATA
path_to_adult_data = ""
datadf = pd.read_csv(path_to_adult_data, header = None, na_values = ['?'], names = names)
datadf = datadf[features_to_keep]
datadf = datadf.dropna()


#### binary label
datadf["new_label"] = 9999
datadf.ix[datadf["label"] ==" <=50K","new_label"] = 0
datadf.ix[datadf["label"] ==" >50K","new_label"] = 1
del datadf["label"]

#
counts = datadf.groupby(["education"])["new_label"].count()
#remove rare categories
datadf = datadf[~datadf["race"].isin([" Amer-Indian-Eskimo", " Other"])]
datadf = datadf[~datadf["education"].isin([" Preschool"])]

### take sub-population defined by bags
df_bags = datadf[datadf["education"].isin(bags)]
bag_props = df_bags.groupby(["education"])["new_label"].sum()/df_bags.groupby(["education"])["new_label"].count()
bag_props = np.round(bag_props,2)
df_bags.head()
eds = df_bags["education"].tolist()
del df_bags["education"]

### binarize data
data = binarize_df(df_bags)



########### 20-fold CV, within each fold generate bags on training
### bag_list -- dictionary mapping row indices to bags
### list_sizes_props -- dictionary giving for each bag its size and label proportion

kf = KFold(len(df_bags), n_folds=20)

for train_index, test_index in kf:
    print len(train_index)
    print len(test_index)
   
    train_rows = test_index.tolist()
    test_rows = train_index.tolist()
    
    data_train = data.iloc[train_rows]
    data_test = data.iloc[test_rows]
    
    label_train = data_train["new_label"].tolist()
    label_train = list(np.where(np.array(label_train)==0,-1,1))

    label_test = data_test["new_label"].tolist()
    label_test = list(np.where(np.array(label_test)==0,-1,1))
    
    ed_train_rows = np.array(eds)[train_rows].tolist()
    bag_list = defaultdict(list)
    for row in range(len(ed_train_rows)):
        ed = ed_train_rows[row]
        bag_list[ed].append(row)
    list_sizes_props ={}
    for bag_name in bag_list.keys()[0:100]:
        b_size = counts[counts.index==bag_name][0]
        b_prop = bag_props[bag_props.index==bag_name][0]
        if b_prop<1:
            list_sizes_props[bag_name] = {b_size:b_prop}
