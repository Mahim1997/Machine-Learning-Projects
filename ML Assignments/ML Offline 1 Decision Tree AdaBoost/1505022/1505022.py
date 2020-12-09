#!/usr/bin/env python
# coding: utf-8

# # 1505022 [ML Offline 1]

# ## Import statements

# In[16]:


import pandas as pd
import numpy as np
import math
import random
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.model_selection import train_test_split
from pprint import pprint
from sklearn.metrics import confusion_matrix


# ### Load datasets

# In[15]:


file_name_dataset_1 = "F:/Programs C and Java/Sessional Things/Assignments-Github/ML Assignments/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
file_name_dataset_2_train = "F:/Programs C and Java/Sessional Things/Assignments-Github/ML Assignments/adult.data"
file_name_dataset_2_test = "F:/Programs C and Java/Sessional Things/Assignments-Github/ML Assignments/adult.test"
file_name_dataset_3 = "F:/Programs C and Java/Sessional Things/Assignments-Github/ML Assignments/creditcardfraud/creditcard.csv"


# ### Entropy calculation

# In[14]:


"""
    Input: examples [numpy array, WITHOUT the labels], labels/classes [numpy array format]
    Output: Entropy of THIS node
    *** epsilon_small is used for log_2 operations (log2(0) may give unwanted exceptions)
    Calculation: 
        for each label x of Labels:
            probability[x] = x/num_examples
        Entropy(node) = H(node) = - [ sum of probability[x]*log_2(probability[x]) ]
"""
def calculate_entropy(examples, labels, epsilon_small = 0.0000000000000000001): # WORKING
    labels_unique = np.unique(labels) # obtain the unique labels of the data
    # gives unique label_names, and counts for each unique label_names
    label_names, label_counts = np.unique(labels,
                                         return_counts = True) 
    label_probabilities = label_counts/sum(label_counts)
    label_log_probabilities = np.log2((label_probabilities + epsilon_small))
    label_products = label_probabilities * label_log_probabilities
    entropy = -1 * sum(label_products)
    if entropy == 0.0:  # to not return -0.0
        entropy = 0.0
    return entropy


# In[13]:


"""
    Finds the feature types ...
"""
# If num of unique vals are greater than 20, declare as continuous [if float/int]
def check_data_type_one(unique_vals, threshold_cnt):  
    if len(unique_vals) == 0:
        return "CATEGORICAL"  # just checking
    sample_val = unique_vals[0]
    if isinstance(sample_val, str):
        return "CATEGORICAL"
    if isinstance(sample_val, float):
        return "CONTINUOUS"
    if len(unique_vals) > threshold_cnt:
        return "CONTINUOUS"
    else:
        return "CATEGORICAL"
# If FLOAT continuous, ELSE if num of unique vals are greater than 20/threshold -> continuous
def find_data_types(data_frame, threshold=20):
    data_type_list = []
    for feature_test in data_frame.columns.values:
        val_first = data_frame[feature_test]
        unique_vals = np.unique(data_frame[feature_test])
        type_val = check_data_type_one(unique_vals, threshold)
        data_type_list.append(type_val)
        # Keep looping
        
    return data_type_list


# ###### Information Gain Calculation wrt one feature_column

# In[12]:


"""
    Input: X (examples), Y (labels), feature_column(idx on X), feature_type [CONTINUOUS/CATEGORICAL],
            use_custom_columns {optional}, custom_columns_list {optional/=}
    Output: Information Gain wrt that feature [CATEGORICAL/BINARIZED]
            Dictionary of information gains wrt each values of that feature [CONTINUOUS]
    Dependency: Uses calculate_entropy() function written above
    Calculation:
        for each value v of examples[feature_column]:
            Partition new_examples[v] by choosing that feature.val == v [if categorical]
            Partition new_examples[v] by choosing that feature.val <= v [if continuous]
            calculate entropy of new_examples, H(S_feature_val_v)
            calculate num_examples(S_feature_val_v)/num_examples(parent_node)
            Use formula IG = H(S{parent}) - [Sum of |S_val|/|S| * H(S_val)]
    Treat either as continuous, or categorical [binarized data is treated as categorical]
"""
def calculate_information_gain(X, Y, feature_column, feature_type,
                              use_custom_columns = False, custom_columns_list = None):  # WORKING
    entropy_parent_node = calculate_entropy(X, Y) # entropy of parent node
    # Either use ALL values for this column OR use custom values only
    if ((use_custom_columns == True)) :
        unique_vals_features = custom_columns_list
    else:
        unique_vals_features = np.unique(X[:, feature_column])
    if feature_type == "CONTINUOUS":
        info_gain_dict = {}
        for val in unique_vals_features:
            idx_left_bool = X[:, feature_column] < float(val)
            idx_right_bool = X[:, feature_column] >= float(val)
            data_left = X[idx_left_bool]
            label_left = Y[idx_left_bool]
            data_right = X[idx_right_bool]
            label_right = Y[idx_right_bool]
            entropy_left = calculate_entropy(data_left, label_left)
            entropy_right = calculate_entropy(data_right, label_right)
            info_gain = entropy_parent_node - (
                ((len(data_left)/len(X)) * entropy_left) + 
                ((len(data_right)/len(X)) * entropy_right)
            )
            info_gain_dict[val] = info_gain
        return info_gain_dict
    else:# CATEGORICAL
        ## Partition into FOR EACH FEATURE
        num_examples_parent = len(X)
        cumulative_entropy = 0.0 # cumulative entropy for all features
        if num_examples_parent == 0: # SOMEHOW comes down to this
            print("-->>Inside calculateInfoGain() .. num_examples_parent = ", num_examples_parent,
                 " returning 0")
            return 0
        for val in unique_vals_features:
            idx_equal_to_feature = X[:, feature_column] == val
            data_of_feature = X[idx_equal_to_feature]
            label_of_feature = Y[idx_equal_to_feature]
            entropy_of_feature = calculate_entropy(data_of_feature, label_of_feature)
            proportion_of_examples_in_feature = float(len(data_of_feature)) / float(num_examples_parent)
            cumulative_entropy = cumulative_entropy + (proportion_of_examples_in_feature * entropy_of_feature)
        # now subtract from parent's entropy to return the information gain
        info_gain = entropy_parent_node - cumulative_entropy
        return info_gain


# In[11]:


"""
    Obtains the best Info-Gain values of THIS GIVEN FEATURE COLUMN [can use custom columns list]
    Input: Dataset (X, Y), feature-column, custom_columns [to NOT consider ALL values for splitting]
    Output: Best Information Gain wrt THIS feature column
    Dependency: Uses calculate_information_gain(X, Y, feature_column, feature_type) function
                which returns the info_gain [if categorical]
                which returns dictionary of {key = split_val, value = info_gain}
"""
def get_best_IG_val_of_this_feature(X_train, Y_train, feature_col, custom_col_list=None,
                                   use_custom_col_list = False):
    if use_custom_col_list == False:
        dict_info_gain_col = calculate_information_gain(X_train, Y_train, 
                  feature_col, feature_type="CONTINUOUS", use_custom_columns=False)
    else:
        dict_info_gain_col = calculate_information_gain(X_train, Y_train, 
  feature_col, feature_type="CONTINUOUS", use_custom_columns=True, custom_columns_list=custom_col_list)
    v = list(dict_info_gain_col.values())
    k = list(dict_info_gain_col.keys())
    return k[v.index(max(v))], max(v) # returns the max gain and index/split_val of that max IG


# In[10]:


"""
    To binarize the data, instead of comparing values for EACH continuous value,
    we will divide into 100 (or user defined number of) data points, 
    and compare with each of those values. [To make it time-efficient (Takes ~ 2mins)]
"""
def get_dictionary_of_best_split_values_for_each_col(X, Y, num_values_of_custom_cols = 100,
                        use_custom_col_list = False, custom_cols = None, use_custom_end_points = False,
                                                    which_cols_to_use_endpoints = None):
    _, ncol = X.shape # Only need the no. of columns
    num_cols_to_do = np.arange(ncol)
    if use_custom_col_list == True:
        num_cols_to_do = custom_cols
    dict_best_ig_feature_split_values = {}  # Dictionary to store {feature_col: split_val, max_IG} that has max IG wrt that feature column 
    for idx_col in range(len(num_cols_to_do)): # EITHER ONLY DO FOR SELECTED COLUMNS/ OR FOR ALL COLUMNS
        col_feature = num_cols_to_do[idx_col]
        unique_vals_of_this_feature = np.unique(X[:, col_feature])
        left_range = int(np.ceil(min(unique_vals_of_this_feature)))
        right_range = int(np.floor(max(unique_vals_of_this_feature))) 
        cols_custom = np.linspace(left_range, right_range, num_values_of_custom_cols)
        if use_custom_end_points == True:
            if which_cols_to_use_endpoints is not None:
                split_val_for_max_IG, max_IG = get_best_IG_val_of_this_feature(X, Y, col_feature, 
                                                       cols_custom, use_custom_col_list=True)
            else:
                if which_cols_to_use_endpoints[idx_col] == True:
                    split_val_for_max_IG, max_IG = get_best_IG_val_of_this_feature(X, Y, col_feature, 
                               cols_custom, use_custom_col_list=True)
        else:
            # ELSE: Use ALL values for continuous ....
            split_val_for_max_IG, max_IG = get_best_IG_val_of_this_feature(X, Y, col_feature)
        dict_best_ig_feature_split_values[col_feature] = split_val_for_max_IG, max_IG
        print("Dictionary: Done for Column = ", col_feature, " split-val = ", split_val_for_max_IG, " Max IG = ", max_IG)
    return dict_best_ig_feature_split_values


# In[9]:


def get_new_features_list(features_list_binarized, features_list_previous_arr):
    features_list_new = {}
    for itr in range(len(features_list_previous_arr)):
        features_list_new[itr] = features_list_previous_arr[itr]
    for key in list(features_list_binarized.keys()):
        features_list_new[key] = features_list_binarized[key]
    return list(features_list_new.values()) # returns as list
"""
    Binarizes the Data wrt the above dictionary found.
    Also, label encoding is done.
"""
def binarize_dataset(X, dict_split_values_and_max_IG_per_col):
    X_bin = X
    keys_list = list(dict_split_values_and_max_IG_per_col.keys())
    features_column = {}
    for col in keys_list:
        # dictionary[col][0] gives split-value and dictionary[col][1] gives max IG.
        split_value_of_col = dict_split_values_and_max_IG_per_col[col][0]
        binarized_data_this_col = X[:, col] < split_value_of_col
        X_bin[:, col] = binarized_data_this_col
        features_column[col] = "BINARIZED"
    
    return X_bin, features_column

def label_encode_labels(Y): # Label encoding of labels/classes/Y ...
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y= le.transform(Y)
    return Y 

def label_encode_data(X, feature_types): # Label encoding of examples/X .... 
    le = preprocessing.LabelEncoder()
    for i in range(len(feature_types)):
        if feature_types[i] == "CATEGORICAL":
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])
    return X

"""
    Input: dataframe
    Output: X, Y [numpy format]
"""
def separate_labels_and_features(df):
    X = df.drop("Label", axis = 1)
    Y = df["Label"]
    return X.values, Y.values


# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


# # https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
# import matplotlib.pyplot as plt
# # The more concentrated ... better it is to binarize
# # For binarization: columns = []
# # dictionary_bins = {0:4, 1:3, 2:8, 3:8, 4:4, 5:3, 6:4, 7:4, 8:5, 9:5, 10:5, 11:5, 12:5, 13:5, 14:5, 15:5, 16:5, 17:5, 18:5, 19:5, 20:5, 21:5, 22:5, 23:5, 24:5, 25:5, 26:5, 27:5, 28:5}

# col = 8
# x,y = np.unique(X[:, col], return_counts=True)
# print(x[0], " ", x[-1]) 
# plt.plot(x, y)
# plt.xlabel('Unique values') 
# plt.ylabel('How many values') 
# plt.title('Dataset 3: Column = ' + str(col))
# plt.show() 


# In[ ]:





# In[ ]:





# In[87]:


"""
Dataset 1: churn dataset
1. Tenure is continuous
2. MonthlyCharges is continuous
3. TotalCharges is continuous [object type ... changed to float type] [Some missing values ... spaces, delete those rows]
4. Drop customerID column
"""
def preprocess_dataset_1(df):    
    df_copy = df.copy(deep = True)
    df_copy.rename(columns = {'Churn' : 'Label'}, inplace=True)
    df_copy = df_copy.drop("customerID", axis = 1)  # drop customer ID
    df_copy.drop(df_copy[df_copy.TotalCharges == ' '].index, inplace=True)  # delete rows with spaces
    df_copy["TotalCharges"] = df_copy["TotalCharges"].astype(float)
    return df_copy

"""
Dataset 3: Credit-Card, all are continuous values, so we will binarize them
Drop the 'time' column, keep only 20,000 NO/False labels and keep ALL YES/True labels
"""
def preprocess_dataset_3(df, data_size = 20000):
    df.rename(columns={'Class':'Label'}, inplace=True)  
    df = df.drop("Time", axis = 1) # drop 'Time' column
    df_yes = df[df['Label'] == 1]
    df_no  = df[df['Label'] == 0]
    indices = df_no.index.tolist()
    test_indices = random.sample(population=indices, k=data_size)  # only keeps 'k' amount of data
    df_no_kept = df.loc[test_indices]
    df = pd.concat([df_yes, df_no_kept]) # recombine the 'YES' and 'NO' samples together into a new dataframe
    return df

def obtain_for_dataset_1(): # dataset 1 by default
    """
        Dataset 1
    """
    print("Inside obtain_for_datasets_1 .. ")

    file_name_dataset = file_name_dataset_1
    print("Reading from file: ", file_name_dataset)
    data_frame_original = pd.read_csv(file_name_dataset) 
    print(data_frame_original.head(2))
    data_frame = preprocess_dataset_1(data_frame_original)
    print(data_frame.head(2))
    
    df_2 = data_frame.drop(data_frame.columns.values[-1], axis=1)
    # print(df_2.head(2))
    feature_types_before = find_data_types(df_2)
    print(feature_types_before, "\n", len(feature_types_before))
    indices_continous = []
    for i in range(len(feature_types_before)):  ## ONLY BINARIZE THE CONTINUOUS FEATURES ... [float/num unique values > 15]
        if feature_types_before[i] == "CONTINUOUS":
            indices_continous.append(i)
    print(indices_continous)
    X, Y = separate_labels_and_features(data_frame) # Separate the labels and features ...
    print("================ BEFORE Binarization ======================")
    dict_split_vals_for_binarization = get_dictionary_of_best_split_values_for_each_col(X, Y, 100,
                                            use_custom_col_list=True, custom_cols=indices_continous)

    print(dict_split_vals_for_binarization)
    print("-------------------- AFTER -------------------------")
    X_binarized, features_col_new = binarize_dataset(X, dict_split_vals_for_binarization)
    features_col_new = get_new_features_list(features_col_new, feature_types_before)
    X_binarized = label_encode_data(X_binarized, features_col_new)
    Y = label_encode_labels(Y)
    feature_types = features_col_new
    print(feature_types)
    # 20 % threshold, random_state = 8, train_test_split using scikitlearn
    X_train, X_test, Y_train, Y_test = train_test_split(X_binarized, Y, test_size=0.2, random_state=8)
    print(X_train.shape, " ", Y_train.shape , " ", X_test.shape, " ", Y_test.shape)
    return X_train, Y_train, X_test, Y_test
    # Pass this binarized data into the engine


def obtain_for_dataset_3(): # dataset 1 by default
    """
        Dataset 3
    """
    print("Inside obtain_for_datasets_3()")
    file_name_dataset = file_name_dataset_3
    print("Reading from file: ", file_name_dataset)
    data_frame_original = pd.read_csv(file_name_dataset) 
    print(data_frame_original.head(2))
    data_frame = preprocess_dataset_3(data_frame_original)
    print(data_frame.head(2))
    
    df_2 = data_frame.drop(data_frame.columns.values[-1], axis=1)
    # print(df_2.head(2))
    feature_types_before = find_data_types(df_2)
    print(feature_types_before, "\n", len(feature_types_before))
    indices_continous = []
    for i in range(len(feature_types_before)):  ## ONLY BINARIZE THE CONTINUOUS FEATURES ... [float/num unique values > 15]
        if feature_types_before[i] == "CONTINUOUS":
            indices_continous.append(i)
    print(indices_continous)
    X, Y = separate_labels_and_features(data_frame) # Separate the labels and features ...
    print("================ BEFORE Binarization/Discretization ======================")
    WHICH_TO_BINARIZE = "ALL"
    if WHICH_TO_BINARIZE == "ALL":
        print("++>> BINARIZING FOR columns = ", indices_continous)
        dict_split_vals_for_binarization = get_dictionary_of_best_split_values_for_each_col(X, Y, 100,
            use_custom_col_list=True, custom_cols=indices_continous)
        print(dict_split_vals_for_binarization)
        X_binarized, features_col_new = binarize_dataset(X, dict_split_vals_for_binarization)
        features_col_new = get_new_features_list(features_col_new, feature_types_before)
        X_binarized = label_encode_data(X_binarized, features_col_new)
        feature_types = features_col_new
        Y = label_encode_labels(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X_binarized, Y, test_size=0.2, random_state=8)
        return X_train, Y_train, X_test, Y_test
    elif WHICH_TO_BINARIZE == "NONE":
        print("-->>NOT BINARIZING ... any columns ... only discretizing")
        indices_continous = []
        num_bins = []  # Try to obtain from the graphs above ...
        dictionary_bins = {0:4, 1:3, 2:8, 3:5, 4:5, 5:5, 6:5, 7:5, 8:5, 9:5, 10:5, 11:5, 12:5, 13:5, 14:5, 15:5, 16:5, 17:5, 18:5, 19:5, 20:5, 21:5, 22:5, 23:5, 24:5, 25:5, 26:5, 27:5, 28:5}
        for col in range(X.shape[1]):  # for each column of X
            num_bins = dictionary_bins[col]  # NOT USING
            # custom no. of bins doesn't give good accuracy
            enc = sklearn.preprocessing.KBinsDiscretizer(n_bins=4, 
                                             encode='ordinal', strategy='kmeans') # custom # bins
            # strategies: quantile, kmeans, uniform [encod = 'ordinal' FOR integer]
            X[:, col] = (enc.fit_transform(X[:, 2].reshape(-1, 1))).reshape(-1, )
        # DISCRETIZATION doesn't require label encoding
        print("-------------------- Returning -------------------------")   
        Y = label_encode_labels(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)
        return X_train, Y_train, X_test, Y_test
    else:  # WHICH_TO_BINARIZE == "SOME"
        print("-->>BINARIZING some columns , discretizing others")
        indices_continous = [7, 28]
        print("++>> BINARIZING columns indices = ", indices_continous)
        dict_split_vals_for_binarization = get_dictionary_of_best_split_values_for_each_col(X, Y, 100,
            use_custom_col_list=True, custom_cols=indices_continous)
        print(dict_split_vals_for_binarization)
        X_binarized, features_col_new = binarize_dataset(X, dict_split_vals_for_binarization)
        features_col_new = get_new_features_list(features_col_new, feature_types_before)
        X_binarized = label_encode_data(X_binarized, features_col_new)
        feature_types = features_col_new
        for col in range(X.shape[1]):  # for each column of X
            if col not in indices_continous:
                enc = sklearn.preprocessing.KBinsDiscretizer(n_bins=8, 
                                                 encode='ordinal', strategy='kmeans') # custom # bins
                X_binarized[:, col] = (enc.fit_transform(X_binarized[:, 2].reshape(-1, 1))).reshape(-1, )
        print("-------------------- Returning -------------------------")   
        Y = label_encode_labels(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X_binarized, Y, test_size=0.2, random_state=8)
        return X_train, Y_train, X_test, Y_test
    
    
    # Pass this binarized data into the engine


# In[369]:


"""
Dataset 2: Adult
[Col, #Unique vals, #All vals]
[ 0 ,  73 ,  32561 ] [ 2 ,  21648 ,  32561 ] [ 4 ,  16 ,  32561 ] 
[ 10 ,  119 ,  32561 ] [ 11 ,  92 ,  32561 ] [ 12 ,  94 ,  32561 ]
col = 2 NEEDS to be discretized, otherwise will take too much time !!
Problems: Rows contain '?', Continuous values for age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
Columns that have '?' marks are : workclass, occupation , native-country [we replace with MODE]
Col =  workclass , Mode =   Private , Percent =  69.70301894904948  %
Col =  occupation , Mode =   Prof-specialty  , Percent =  12.714597217530175  %
Col =  native-country , Mode =   United-States  , Percent =  89.5857006848684  %
"""
def preprocess_dataset_2_train(df_original):
    df = df_original.copy(deep = True)
    print("No. of columns = ", len(df.columns.values))
    df['education-num'] = df['education-num'].astype(float)  # to make THIS column float [so that it automatically becomes continuous]
    list_columns_with_QUESTION_mark = []
    for col in df.columns.values:
        unique_vals, unique_counts = np.unique(df[col].values, return_counts=True)
        missing_val = " ?"
        if missing_val in unique_vals:  # ONLY THOSE COLUMNS THAT CONTAIN THE ' ?' mark
            list_columns_with_QUESTION_mark.append(col)
            mode_of_unique_vals = unique_vals[np.argmax(unique_counts)]
#             print("Col = ", col, " Mode = ", mode_of_unique_vals, " Percent = ", (unique_counts[np.argmax(unique_counts)]/len(df))*100, " %")
            df[col] = np.where((df[col] == ' ?'),mode_of_unique_vals,df[col])  # Replace '?' with mode of that column 
    return df, list_columns_with_QUESTION_mark

def obtain_train_test_for_dataset_2():
    column_names_dataset_2 = ["age", "workclass", "fnlwgt", "education",
                       "education-num", "marital-status", "occupation",
                       "relationship", "race", "sex", "capital-gain",
                       "capital-loss", "hours-per-week", "native-country", "Label"]
    for col in column_names_dataset_2:
        print(col, end = ',')
    print("-->> *** NEED to add the above list as HEADER", col)
    print("Train file: ", file_name_dataset_2_train, " Test file: ", file_name_dataset_2_test)
    data_frame_original = pd.read_csv(file_name_dataset_2_train)
    data_frame, list_col_with_qstn_mark = preprocess_dataset_2_train(data_frame_original)
    data_frame_test_original = pd.read_csv(file_name_dataset_2_test)
    data_frame_test, list_col_with_qstn_mark_test = preprocess_dataset_2_train(data_frame_test_original)
    X_train = (data_frame.drop("Label", axis = 1)).values
    Y_train = (data_frame["Label"]).values
    X_test = (data_frame_test.drop("Label", axis = 1)).values
    Y_test = (data_frame_test["Label"]).values
    print("Train: ", X_train.shape, " ", Y_train.shape, "  Test: ", X_test.shape, " ", Y_test.shape)
    
    df_2 = data_frame.drop(data_frame.columns.values[-1], axis=1)
    feature_types_before = find_data_types(df_2)
    print(feature_types_before, "\n", len(feature_types_before))
    indices_continous = []
    for i in range(len(feature_types_before)):  ## ONLY BINARIZE THE CONTINUOUS FEATURES ... [float/num unique values > 15]
        print(df_2.columns.values[i], " ", feature_types_before[i])
        if feature_types_before[i] == "CONTINUOUS":
            indices_continous.append(i)
    print(indices_continous)
    print("================ BEFORE Binarization ======================")
    indices_continous.remove(2) # REMOVE "fnwlgt" column [will discretize it later]
    
    dict_binarized_values = get_dictionary_of_best_split_values_for_each_col(X_train, Y_train, 100,
                                use_custom_col_list=True, custom_cols=indices_continous,
                                use_custom_end_points=False)

    print(dict_binarized_values)
    print("-------------------- AFTER -------------------------")
    X_binarized, features_col_new = binarize_dataset(X_train, dict_binarized_values) # binarize
    features_col_new = get_new_features_list(features_col_new, feature_types_before)
    X_train = label_encode_data(X_binarized, features_col_new) # label encoding [EXAMPLES, Training data]
    enc = sklearn.preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans') # 10 bins
    # strategies: quantile, kmeans, uniform [encod = 'ordinal' FOR integer]
    X_train[:, 2] = (enc.fit_transform(X_train[:, 2].reshape(-1, 1))).reshape(-1, )
    Y_train = label_encode_labels(Y_train) # label encoding [LABELS, Training data]
    X_test, features_col_new_test = binarize_dataset(X_test, dict_binarized_values) # binarize
    X_test = label_encode_data(X_test, features_col_new) # label encoding [EXAMPLES, TEST DATA]
    X_test[:, 2] = (enc.fit_transform(X_test[:, 2].reshape(-1, 1))).reshape(-1, )  # DISCRETIZATION
    Y_test = label_encode_labels(Y_test) # label encoding [LABELS, TEST DATA]
    feature_types = features_col_new
    print(feature_types)
    return X_train, Y_train, X_test, Y_test


# ### DTreeNode and DTreeClassifier as classes

# In[50]:


"""
    Class DTreeNode to store the decision trees/subtrees nodes.
    CAN'T HAVE CONTINUOUS DATA [ONLY BINARIZED/CATEGORICAL(Pre-existing or Discretized)]
"""
class DTreeNode:
    def __init__(self):
        self.children = {} # [feature_val: DTreeNode]
        self.feature_col = -1 # which feature column THIS node contains as a question
        self.is_leaf_node = False # by default, shouldn't be a leaf
        self.classification = "NONE" # also consider as the plurality value
        
    def printTree(self, spaces_num = 0):
        node = self
        if node.is_leaf_node == True:  # Only print the classification column/feature
            print(" " * spaces_num, "Lab(", node.classification, ")")
            return
        else:  # Print the question
            print("\n", ("    " * spaces_num), "Q(", node.feature_col, ")")
        spaces_num = spaces_num + 1
        for key in list(self.children.keys()):
            print("  "*spaces_num, " == ", key)
            node = self.children[key]
            node.printTree(spaces_num)
            


# In[51]:


"""
    Prints various metrics.
"""
def print_metrics(Y_true, Y_predicted):
    TN, FP, FN, TP = confusion_matrix(Y_true, Y_predicted).ravel()
    accuracy = (TP + TN)/(TP+TN+FP+FN)
    recall = (TP)/(TP + FN)
    specificity = (TN)/(TN + FP)
    precision = (TP)/(TP + FP)
    false_discovery_rate = (FP)/(TP + FP)
    f1_score = 2*((precision * recall) / (precision + recall))
    print("TN = ", TN, " FP = ", FP, " FN = ", FN, " TP = ", TP)
    print("Accuracy = ", accuracy*100, "%")
    print("TPR = Sensitivity = Recall = ", recall*100, "%")
    print("TNR = Specificity = ", specificity*100, "%")
    print("Precision = PPV = Positive Predictive Value = ", precision*100, "%")
    print("FDR = False Discovery Rate = ", false_discovery_rate*100, "%")
    print("F1 Score = ", f1_score*100, "%")


# In[52]:


"""
    Can't have CONTINUOUS data [SHOULD be binarized/discretized first]
    Decision Tree Classifier
"""
def return_majority_label(labels): # Returns the majority of the label
    # obtains unique labels AND also the counts of those unique labels
    label_names, label_counts = np.unique(labels, return_counts = True)
    index_max = label_counts.argmax()
    labels_with_max_count = label_names[index_max]
    return labels_with_max_count

class DTreeClassifier:
    def __init__(self):
        self.d_tree_root = DTreeNode()
        self.max_depth = 5  # default max-depth = 5
        self.print_termination_msg = True
    
    def form_a_leaf_node(self, labels):
        leaf_node = DTreeNode()
        leaf_node.is_leaf_node = True
        leaf_node.classification = return_majority_label(labels)
        return leaf_node
    ##### Can't have CONTINUOUS data [ONLY CATEGORICAL/BINARIZED DATA IS ALLOWED !!]
    def recursive_fit(self, X, Y, X_parent, Y_parent, current_depth, max_depth):
        ## 1. If current-depth == max-depth [Base Case]
        if current_depth == max_depth:
            return self.form_a_leaf_node(Y) # return the leaf node with majority of the current labels
        ## 2. If EXACTLY one label ... return that
        if (len(np.unique(Y)) == 1):
            return self.form_a_leaf_node(Y) # return the leaf node with majority of the current labels
        ## 2_2. If no more examples, return max label of parent
        if len(X) == 0:
            return self.form_a_leaf_node(Y_parent)
        
        current_depth = current_depth + 1  # increment current_depth variable
        val_max_IG = col_max_IG = -1
        for col in range(0, X.shape[1]):  ## for each number of columns/features
            ig_this_col = calculate_information_gain(X, Y, col, "CATEGORICAL") # DOESN't use feature_types_array
#             print("col = ", col, " ig_col = ", ig_current_col)
            if ig_this_col > val_max_IG:
                val_max_IG = ig_this_col
                col_max_IG = col
        
        ## 3. Max IG obtained is 0 [no more examples/features]
        if val_max_IG == 0:
            return self.form_a_leaf_node(Y)
        
        d_tree_node = DTreeNode()
        d_tree_node.classification = return_majority_label(Y)  ### Saves the pluarility value of THIS node [used in prediction]
        ## -> Recursion
        for val_of_this_feature in np.unique(X[:, col_max_IG]):
            index_child = (X[:, col_max_IG] == val_of_this_feature)
            X_child = X[index_child]
            Y_child = Y[index_child]
            d_tree_node.is_leaf_node = False  ### is an internal node
            d_tree_node.feature_col = col_max_IG  ### question is to be asked on this column/feature
            sub_tree = self.recursive_fit(X_child, Y_child, X, Y, current_depth, max_depth)
            d_tree_node.children[val_of_this_feature] = sub_tree
        ### END OF FOR LOOP, return d_tree_node
        return d_tree_node
    
    def printTree(self):
        self.d_tree_root.printTree()

    def fit(self, examples, labels): # FIT FUNCTION [calls recursive_fit function above using suitable params]
        dt_node = self.recursive_fit(examples, labels, examples, labels,
                                     current_depth = 0, max_depth=self.max_depth)
        self.d_tree_root = dt_node
        if self.print_termination_msg == True:
            print("Fit done for, max-depth = ", self.max_depth)
        
    def predict_one_example(self, example_to_predict):
        self.d_tree_root_backup = self.d_tree_root
        root = self.d_tree_root
        if root == None:
            raise Exception('Root of the Decision Tree is null !! [In predict()]')
        while root is not None:
            if root.is_leaf_node == True: # classify ... since, it is the leaf
                self.d_tree_root = self.d_tree_root_backup
                return root.classification
            else:
                col_to_process = root.feature_col
                val_present_in_example = example_to_predict[col_to_process]
                bool_found = False
                keys_feature_vals = list(root.children.keys())
                for feature_val in keys_feature_vals:
                    if example_to_predict[root.feature_col] == feature_val:
                        root = root.children[feature_val]
                        bool_found = True
                        break
                    
                if bool_found == False: ### SHOULD RETURN PARENT's PLURALITY VALUE
                    self.d_tree_root = self.d_tree_root_backup
                    return root.classification # PARENT's plurality value
                    #raise Exception("The value of ", val_present_in_example, " doesn't exist for col_idx = ", col_to_process)
                
    def predict(self, examples_test):
        labels = []
        for example in examples_test:
            yp1 = self.predict_one_example(example)
            labels.append(yp1)
        return labels


# In[85]:


"""
    Obtain X_train, Y_train, X_test, Y_test wrt which dataset.
"""
# X_train, Y_train, X_test, Y_test = obtain_for_dataset_1()
# X_train, Y_train, X_test, Y_test = obtain_train_test_for_dataset_2()
X_train, Y_train, X_test, Y_test = obtain_for_dataset_3()
print("-->>Training: ", X_train.shape, " ", Y_train.shape, "   TESTING: ", X_test.shape, " ", Y_test.shape)


# #### Script to train and predict using Decision Tree

# In[86]:


t_before = datetime.datetime.now()
d_tree_custom_DT = DTreeClassifier()
# d_tree.fit(X_train[0:100], Y_train[0:100], max_depth = 10000)
d_tree_custom_DT.max_depth = 10000 # here, depth-max is not to be considered ...
d_tree_custom_DT.fit(X_train, Y_train)
Y_pred_train = d_tree_custom_DT.predict(X_train)
print("------------- Training ----------------")
print_metrics(Y_train, Y_pred_train)
Y_pred_test = d_tree_custom_DT.predict(X_test)
print("------------- Testing ----------------")
t_after = datetime.datetime.now()
del_t = t_after - t_before
print_metrics(Y_test, Y_pred_test)

print("\n\nTotal time taken = ", del_t.seconds, " s")


# In[57]:


"""
AdaBoost Class:
h: contains the K number of classifiers
z: contains the weights of each classifier
K : number of classifiers   
w, a vector of 'N' weights, 1/N initially
L_weak = self.produce_classifier_generic() # generic interface [in this case a decision stump]
"""
class AdaBoost:
    def __init__(self, mode="DECISION_STUMP"):
        self.mode = mode
    
    # factory method to produce classifier
    def produce_classifier_generic(self): # for now, default is decision stump
        if self.mode == "DECISION_STUMP":
            classifier = DTreeClassifier()
            classifier.print_termination_msg = False
            classifier.max_depth = 1 # decision stump
        elif self.mode == "SKLEARN":
            classifier = DecisionTreeClassifier(criterion='entropy', max_depth = 1) # DOESN'T work
        else:  # Other modes depending on the 'mode' variable
            classifier = None
        return classifier
    
    
    def normalize(self, w):
        if sum(w) == 0:
            return w
        else:
            return w/sum(w)
    
    def resample(self, examples, labels, w):
        # returns resampled data and labels according to weights
        w = self.normalize(w)
        resampled_data_size = len(labels)
        temp_arr = np.arange(resampled_data_size)
        indices_sampled = np.random.choice(temp_arr, len(temp_arr), p=w)  # N resampled data points for now
        examples_sampled = []
        labels_sampled = []
#         print("Inside resample ... indices_sampled = ", indices_sampled)
        for idx in indices_sampled:  # probably a better method exists using numpy
            examples_sampled.append(examples[idx])
            labels_sampled.append(labels[idx])
        
        return np.asarray(examples_sampled), np.asarray(labels_sampled)
    
    # Return the label with the maximum weight of classifier
    def predict(self, example):
        labels_hypothesis = np.zeros(self.K) # Initially set to zeros
        unique_labs_weights = {}
        for k in range(self.K):
            labels_hypothesis[k] = self.h[k].predict_one_example(example)            
        for unique_labels in (np.unique(labels_hypothesis)):
            unique_labs_weights[unique_labels] = 0.0 # initially set as 0 weight
        for k in range(self.K):
            unique_labs_weights[labels_hypothesis[k]] = unique_labs_weights[labels_hypothesis[k]] + self.z[k] 
        
        v = list(unique_labs_weights.values())
        k = list(unique_labs_weights.keys())
        return k[v.index(max(v))]  # return the unique label with the MAX weight
        
        
    def boost(self, examples, labels, K):     
        self.K = K
        N = examples.shape[0]
        self.w = np.full(shape=(N, ), fill_value = (1/N))
        self.h = {} # dictionary
        self.z = np.zeros(shape=(K, ))
        for k in range(K):
            data_resampled, labels_resampled = self.resample(examples, labels, self.w)
#             print("-> k = ", k, " w = ", self.w, "\nlabels_counts = ", np.unique(labels_resampled, return_counts=True))
            self.h[k] = self.produce_classifier_generic()
            # FIT on RESAMPLED data [labels are also resampled]
            self.h[k].fit(data_resampled, labels_resampled) 
#             h[k] = L_weak
            error = 0
            for j in range(0, N):
                if self.h[k].predict_one_example(examples[j]) != labels[j]: # PREDICT on INITIAL EXAMPLES
                    error = error + self.w[j]
#             print("-->>k = ", k, ", Error = ", error, "\n\n")
            if error > 0.5:
                continue
            for j in range(0, N):
                if self.h[k].predict_one_example(examples[j]) == labels[j]:
                    self.w[j] = self.w[j] * (error/(1 - error))  # error CAN'T be 1 due to continue condition
            
            # Normalize w
            self.w = self.normalize(self.w)
            # Error can be 0
            if error == 0:
                self.z[k] = 10 # max of 10^10 ?? [impossible case so, assign a large number]
            else:
                self.z[k] = np.log((1-error)/error)
        # No need to return, since h,z are going to be stored 
#         return self.Weighted_Majority(h, z)
    


# In[58]:


"""
AdaBoost: running script
"""
k_list = [5, 10, 15, 20]
# k_list = [5]
t_before = datetime.datetime.now()
for K in k_list:
    adaBoost = AdaBoost()
#     adaBoost = AdaBoost("SKLEARN") # To check
    adaBoost.boost(X_train, Y_train, K)
    Y_pred_train_adaBoost = []
    for ex in X_train:  # predict train dataset
        Y_pred_train_adaBoost.append(adaBoost.predict(ex))
    Y_pred_test_adaBoost = []
    for ex in X_test:  # predict test dataset
        Y_pred_test_adaBoost.append(adaBoost.predict(ex))
    print("------------- Train AdaBoost K = ", K, "-----------------")
    print_metrics(Y_train, Y_pred_train_adaBoost)
    print("------------- Test AdaBoost K = ", K, "-----------------")
    print_metrics(Y_test, Y_pred_test_adaBoost)
    
t_after = datetime.datetime.now()
del_t = t_after - t_before
print("Total time taken = ", del_t.seconds, " s")


# In[59]:


"""
    To compare using Sklearn's Decision Tree Classifier
"""
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, Y_train)
Y_p_train = dtc.predict(X_train)
Y_p_test = dtc.predict(X_test)
print("=============== SKLEARN TRAINING [to compare]=================")
print_metrics(Y_p_train, Y_train)
print("+++++++++++++++ SKLEARN TESTING [to compare]++++++++++++++++++")
print_metrics(Y_p_test, Y_test)

