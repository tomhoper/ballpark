# -*- coding: utf-8 -*-
"""
Feasibility method (Problem 7) for Boston dataset, synthetic constraints

@author: tomhope
"""

import cvxpy as cp
from sklearn.datasets import load_boston
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import operator


def get_all_indices_per_bag(data,bag_names):
    
    bag_indices = {}
    for group_name in bag_names:
        #print group_name
        bag_indices[group_name+"1"] =  list(np.where(data[group_name] == 1)[0])
        bag_indices[group_name+"2"] =  list(np.where(data[group_name] == 2)[0])
        bag_indices[group_name+"3"] =  list(np.where(data[group_name] == 3)[0])

    return bag_indices


def get_upper_lower_constraints_per_bag(data,bag_names,error_term=None):
    
    upper_p_bound_bags = {}
    lower_p_bound_bags = {}
    
    for group_name in bag_names:
        
            constraint_data = data.groupby(by=[group_name]).mean()["target"]        
    
            upper_p_bound_bags[group_name+"1"] =  constraint_data[1]*(1+error_term)
            lower_p_bound_bags[group_name+"1"] =  constraint_data[1]*(1-error_term)
            
            upper_p_bound_bags[group_name+"2"] =  constraint_data[2]*(1+error_term)
            lower_p_bound_bags[group_name+"2"] =  constraint_data[2]*(1-error_term)
            
            upper_p_bound_bags[group_name+"3"] =  constraint_data[3]*(1+error_term)
            lower_p_bound_bags[group_name+"3"] =  constraint_data[3]*(1-error_term)
            
    return upper_p_bound_bags,lower_p_bound_bags
    
def CreateConstraints_compact(list_props):
    sorted_x = sorted(list_props.items(), key=operator.itemgetter(1))[::-1]
    constraints = []
    for j in range(len(sorted_x)-1):
        p0 = sorted_x[j][0]
        p1 = sorted_x[j+1][0]
        constraints.append((p0,p1))
    
    return constraints
    
def feasibility_regression(X, pairwise_constraints_indices, 
                      bag_indices,upper_p_bound_bags,
                      diff_upper_bound_pairs,diff_lower_bound_pairs,
                      lower_p_bound_bags):

    theta = cp.Variable(X.shape[1])
    reg = cp.square(cp.norm(theta, 2))
    
    constraints = []
    added_pairs = []
    pair_ind = 0
    for pair in pairwise_constraints_indices:
        bag_high = bag_indices[pair[0]]
        bag_low = bag_indices[pair[1]]
      
        scores_high = (1./len(bag_high))*X[bag_high]*theta
        scores_low = (1./len(bag_low))*X[bag_low]*theta
    
        if pair in diff_upper_bound_pairs:
            constraints.append(cp.sum_entries(scores_high) - cp.sum_entries(scores_low) < diff_upper_bound_pairs[pair])
            
        if pair in diff_lower_bound_pairs:
            constraints.append(cp.sum_entries(scores_high) - cp.sum_entries(scores_low) > diff_lower_bound_pairs[pair])
        else:
            constraints.append(cp.sum_entries(scores_high) - cp.sum_entries(scores_low) > 0)
    
        if pair[0] not in added_pairs:
            if pair[0] in upper_p_bound_bags:
                constraints.append(cp.sum_entries(scores_high)<=upper_p_bound_bags[pair[0]])
            if pair[0] in lower_p_bound_bags:
                constraints.append(cp.sum_entries(scores_high)>=lower_p_bound_bags[pair[0]])
            added_pairs.append(pair[0])
        if pair[1] not in added_pairs:
            if pair[1] in upper_p_bound_bags:
                constraints.append(cp.sum_entries(scores_low)<=upper_p_bound_bags[pair[1]])
            if pair[1] in lower_p_bound_bags:
                constraints.append(cp.sum_entries(scores_low)>=lower_p_bound_bags[pair[1]])
            added_pairs.append(pair[1])
        pair_ind+=1
    
    prob = cp.Problem(cp.Minimize(1*reg),constraints = constraints)

    try:
        prob.solve()
    except:
        prob.solve(solver="SCS")
    w_t = np.squeeze(np.asarray(np.copy(theta.value)))
    return w_t        

#load data    
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target


train_rows,test_rows = train_test_split(range(len(df)),test_size=0.25,random_state=123)

df_train = df.iloc[train_rows]
df_test = df.iloc[test_rows]


mmin = df_train.min()
mmin["target"] = 0
mmax = df_train.max()
mmax["target"] = 1

df_train -= mmin
df_train /= mmax
df_test -= mmin
df_test /= mmax

y_train = df_train["target"].values
y_test = df_test["target"].values
X_train = df_train.values[:,:13]
X_test = df_test.values[:,:13]


X_train = np.hstack([X_train, np.ones((X_train.shape[0],1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0],1))])
    


#3 features, 9 bags
group_vars = ["RM","CRIM","NOX"]
for var in group_vars:  
    df_train[var+"_group"] = pd.Series(None, index=df.index)
    df[var+"_group"] = pd.Series(None, index=df.index)

#Discretize to create bags
quantiles_dict = {}
for var in group_vars:  
    quantiles_dict[var] = df_train.quantile([0.33,0.66])[var]
    for i in df_train.index:
        if df_train.loc[i][var]<=quantiles_dict[var][0.33]:
           df_train.loc[i][var+"_group"] = 1
        elif (df_train.loc[i][var]>quantiles_dict[var][0.33]) and (df_train.loc[i][var]<=quantiles_dict[var][0.66]):
           df_train.loc[i][var+"_group"] = 2
        elif (df_train.loc[i][var]>quantiles_dict[var][0.66]):
           df_train.loc[i][var+"_group"] = 3
for var in group_vars:  
    quantiles_dict[var] = df.quantile([0.33,0.66])[var]
    for i in df.index:
        if df.loc[i][var]<=quantiles_dict[var][0.33]:
           df.loc[i][var+"_group"] = 1
        elif (df.loc[i][var]>quantiles_dict[var][0.33]) and (df.loc[i][var]<=quantiles_dict[var][0.66]):
           df.loc[i][var+"_group"] = 2
        elif (df.loc[i][var]>quantiles_dict[var][0.66]):
            df.loc[i][var+"_group"] = 3
    
group_vars2 = [g+"_group" for g in group_vars]
bag_indices = get_all_indices_per_bag(data = df_train,bag_names = group_vars2)    



###GET TRUE MEAN PER BAG
bag_means = {}
for bag_name,indices in bag_indices.items():
    bag_means[bag_name] = np.mean(y_train[indices])      


##CREATE UPPER/LOWER WITH MEANS + MULTIPLCATIVE TERM
upper_p_bound_bags, lower_p_bound_bags = get_upper_lower_constraints_per_bag(df_train,group_vars2,
                                                                             error_term=0.1)
    
#CREATE PAIRWISE CONSTRAINTS BASED ON TRUE MEANS
pairwise_constraints_indices = CreateConstraints_compact(bag_means)


#CREATE BAG DIFFERENCE BOUNDS ON MEAN
upper_diff_bound_bags = {}
lower_diff_bound_bags = {}
for pair in pairwise_constraints_indices:
    upper_diff_bound_bags[pair] = 1.1*(bag_means[pair[0]] - bag_means[pair[1]])  


#Get weight vector w    
w_t = feasibility_regression(X = X_train,
      pairwise_constraints_indices = pairwise_constraints_indices,
      bag_indices = bag_indices,upper_p_bound_bags = upper_p_bound_bags,
      diff_upper_bound_pairs = upper_diff_bound_bags,
      diff_lower_bound_pairs = lower_diff_bound_bags,
      lower_p_bound_bags = lower_p_bound_bags)
#eval
pred_test = np.dot(X_test,w_t)
med_err = metrics.median_absolute_error(y_test, pred_test)
print med_err
print np.sqrt(np.mean(np.power(pred_test - y_test,2)))