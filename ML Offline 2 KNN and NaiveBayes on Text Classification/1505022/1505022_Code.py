%%time
### MOUNT DRIVE .....
from google.colab import drive
drive.mount('/content/drive')

# -*- coding: utf-8 -*-
"""ML-Offline-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qf4sNPao72Uvs18qDY9j3SGpIvTZN8Go
"""



## First Import statements
import numpy as np
import pandas as pd
from IPython.display import display
import pickle

from scipy import stats
from xml.dom import minidom

from sklearn.metrics import pairwise_distances, accuracy_score


# !pip3 install tqdm
from tqdm import tqdm
import operator
from functools import reduce

FOLDER_TRAIN = '/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Data/Training/'
# FILE_TOPICS = '/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Data/topics.txt'
FILE_TOPICS = '/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Data/topics-all.txt'

# !ls "$FOLDER_TRAIN"

## Set random seed
RANDOM_STATE = 22
np.random.seed(RANDOM_STATE)

## Read data
topic_remove = '3d_Printer' ## Remove 3D-printer
list_doc_types = []
with open(FILE_TOPICS, 'r') as fp:
    topic = fp.readline()
    while topic:
        topic = topic.replace("\n", "")
        list_doc_types.append(topic)
        topic = fp.readline()

list_doc_types.remove(topic_remove)
print(list_doc_types)

print(list_doc_types)

def get_train_val_test_data(file_name):
    xmldoc = minidom.parse(file_name)
    xml_list = xmldoc.getElementsByTagName('row') ## tag using 'row'
    
    # print(f"Inside get_train_val_test_data(), len(item_list) = {len(item_list)}")

    item_list = [x.attributes['Body'].value for x in xml_list]

    train_list = item_list[0:500] ## first 500 train
    val_list =  item_list[500:700] ## next 200 val
    test_list = item_list[700:1200] ## next 500 test

    ## delete original list.
    del item_list
    del xmldoc

    ## return the new lists
    return train_list, val_list, test_list

def add_to_dataframe(df_old, to_add):
    df_old = df_old.append(pd.Series(to_add, index=df_old.columns), ignore_index=True)
    return df_old

## Create three dataframes.

## https://www.kite.com/python/answers/how-to-create-an-empty-dataframe-with-column-names-in-python
column_names =["content", "Label"]

df_train = pd.DataFrame(columns = column_names)
df_val = pd.DataFrame(columns = column_names)
df_test = pd.DataFrame(columns = column_names)

### Populate all topics
def populate_data_frames(df_train, df_val, df_test, list_doc_types):
    for topic in list_doc_types: ## iterating per topic/label
        label = topic ## assign label
        
        ## read using xml package
        file_name = FOLDER_TRAIN + topic + ".xml"
        xmldoc = minidom.parse(file_name)
        xml_list = xmldoc.getElementsByTagName('row') ## tag using 'row'

        ## get train, val, test lists
        train_list, val_list, test_list = get_train_val_test_data(file_name=file_name)

        print(len(train_list), len(val_list), len(test_list))

        ## add to dataframe
        for v1 in train_list:
            df_train = add_to_dataframe(df_old=df_train, to_add=[v1, label])
        for v2 in val_list:
            df_val = add_to_dataframe(df_old=df_val, to_add=[v2, label])
        for v3 in test_list:
            df_test = add_to_dataframe(df_old=df_test, to_add=[v3, label])


        ## delete original list
        del train_list
        del val_list
        del test_list

    ## return dataframes
    return df_train, df_val, df_test

### Call the function
df_train, df_val, df_test = populate_data_frames(df_train=df_train, df_val=df_val, df_test=df_test, list_doc_types=list_doc_types)

print(f"len df_train = {len(df_train)}")
print(f"len df_val = {len(df_val)}")
print(f"len df_test = {len(df_test)}")

"""## Shuffle dataframes"""

df_train = df_train.sample(frac=1, random_state=RANDOM_STATE)
df_val = df_val.sample(frac=1, random_state=RANDOM_STATE)

"""## Import libraries"""

## https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
## https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
## from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

## https://stackoverflow.com/questions/26693736/nltk-and-stopwords-fail-lookuperror
## https://stackoverflow.com/questions/26570944/resource-utokenizers-punkt-english-pickle-not-found

# nltk.download()
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

"""## Preprocess each sentence"""

## https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
## First lemmatize, and then stem.

def pre_process_stem_sentence(sentence, stop_words, stemmer, lemmatizer):
    ## save in another variable
    text = sentence
    ## remove HTML tags [using soup]
    soup = BeautifulSoup(text)
    text = soup.get_text()
    
    ## remove <a href> type things
    soup = BeautifulSoup(text) ## create soup again.
    for a in soup.findAll('a'):
        a.replaceWithChildren()
    
    text = str(soup) ## reform text


    ## remove unicode.
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r'[-+]?\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]',' ', text)

    ## remove links
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"https\S+", "", text)
    ## text = re.sub(r"www\S+", "", text)

    ## to be safe, remove HTML tags again using regex
    #### https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44  
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)

    ## convert to small letters.
    text = text.lower()

    ## replace newlines, tabs -> SPACE
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    ## {COLON} make <a>:<b> become <a>[space]<b>
    # text = text.replace(": ", ":") # :[space] -> : 
    text = text.replace(":", " ")  # : -> [space]
    
    ## {HYPHEN}
    # text = text.replace("- ", "-")
    text = text.replace("-", " ")

    ## numbers removal    
    text = re.sub(r'[-+]?\d+', '', text)

    ## punctuations removal
    text = text.translate((str.maketrans('','',string.punctuation)))   


    ## [remove anything EXCEPT english letters]
    ## https://stackoverflow.com/questions/6323296/python-remove-anything-that-is-not-a-letter-or-number
    text = re.sub(  "[^a-z ]",              # Anything except 0..9, a..z and A..Z
                    "",                     # replaced with nothing
                    text)                   # in this string   

    ## remove space initially and finally.
    text = text.lstrip()

    ## make double spaces become one space.
    text = re.sub('\s+', ' ', text)

    ## remove stop words (English).
    # print(f"Before stopwords removal, text.len = {len(text)}")

    tokens = word_tokenize(text)
    stop_words_removed_words_list = [t for t in tokens if not t in stop_words]

    # print(f"After tokenize and stopwords, len stop_words_removed_words_list = {len(stop_words_removed_words_list)}")

    ## apply lemmatization.
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stop_words_removed_words_list]
    
    ## apply stemming.
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    ## return the final pre-processed list-of-words.
    return stemmed_words

################ Test on one sentence #######################

## Obtain once.
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

## Preprocess per element of dataframe to test.
text_to_process = df_train['content'].iloc[499]
print(text_to_process)

print("--"*90)

preprocessed_text = pre_process_stem_sentence(sentence=text_to_process, stop_words=stop_words, stemmer=stemmer, lemmatizer=lemmatizer)
print(preprocessed_text)

doc = df_val['content'].iloc[499]
print(doc)

print("--"*85)

preprocessed = pre_process_stem_sentence(sentence=doc, stop_words=stop_words, stemmer=stemmer, lemmatizer=lemmatizer)
print(len(preprocessed))
print(preprocessed)

## Preprocess for each sentence of train-dataframe.
df_train["stemmed_content"] = df_train.apply(lambda row : pre_process_stem_sentence(sentence=row['content'], stop_words=stop_words, stemmer=stemmer, lemmatizer=lemmatizer), axis=1)
df_val["stemmed_content"] = df_val.apply(lambda row : pre_process_stem_sentence(sentence=row['content'], stop_words=stop_words, stemmer=stemmer, lemmatizer=lemmatizer), axis=1)
df_test["stemmed_content"] = df_test.apply(lambda row : pre_process_stem_sentence(sentence=row['content'], stop_words=stop_words, stemmer=stemmer, lemmatizer=lemmatizer), axis=1)

"""# Form Vocabulary from dataframe_train"""

def print_first_n_keys_and_vals_dict(dictionary, n=5):
    print(f"Len dictionary = {len(dictionary)}, printing first {n} keys, vals")
    itr = 0
    for key in dictionary:
        print(key, dictionary[key])
        itr += 1
        if itr == n:
            break

### Also remove one-len words
dictionary_vocab = {} ## empty dict.
for doc in df_train['stemmed_content']:
    # print(f"len doc = {len(doc)}")
    for word in doc:
        if len(word) <= 1:
            continue

        len_currently = len(dictionary_vocab) ## add to len. [idx new]

        if word not in dictionary_vocab:
            dictionary_vocab[word] = (0, len_currently)
        else:
            (val, idx) = dictionary_vocab[word]
            dictionary_vocab[word] = (val + 1, idx) ## Maintain the same index.

print(f"len dictionary_vocab = {len(dictionary_vocab)}")

# file_name = "/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Vocab.txt"
# with open(file_name, 'w') as fw:
#     for voc in dictionary_vocab:
#         fw.write(voc)
#         fw.write("\n")

# ## https://www.kite.com/python/answers/how-to-save-a-dictionary-to-a-file-in-python
# vocab_file = '/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/vocab_dict_df_train.pkl'

# a_file = open(vocab_file, "wb")
# pickle.dump(dictionary_vocab, a_file)
# a_file.close()

# # a_file = open(vocab_file, "rb")
# # dictionary_vocab = pickle.load(a_file)
# # print(dictionary_vocab)

"""## Keep only those docs in train whose lengths are above 3 words i.e.
### length of stemmed content > 3
"""

lens = [len(x) for x in df_train['stemmed_content']]
print(max(lens))
print(min(lens))

print(f"Before removal, len df_train = {len(df_train)}")

MIN_WORD_COUNT_TO_REMOVE = 2
# print(df_train[df_train['stemmed_content'].str.len() <= MIN_WORD_COUNT_TO_REMOVE]["stemmed_content"])
idxToRemove = df_train[df_train['stemmed_content'].str.len() <= MIN_WORD_COUNT_TO_REMOVE].index
print(f"To remove num items = {len(idxToRemove)}")

labels_to_remove = df_train['Label'].iloc[idxToRemove].values
print(np.unique(labels_to_remove, return_counts=True)) ## Almost fairly distributed.


df_train.drop(idxToRemove , inplace=True) ### Removing

print(f"After removal, len df_train = {len(df_train)}")

"""## Create Hamming Distance Vectors by representing with 0/1"""

def form_hamming_vector(list_words, vocab_dict):
    vec = np.zeros(len(vocab_dict)) # +1 for unknown word.
    for word in list_words:
        if word not in vocab_dict: ## add 1 to unkown word index. [DO NOT]
            # vec[UNKNOWN_WORD_INDEX] = 1
            continue
        else: ## word is present in vocab, get index.
            (value, idx) = vocab_dict[word]
            vec[idx] = 1
    return vec

## Create hamming vectors for each col of dataframe
df_train["ham_vector"] = df_train.apply(lambda row : form_hamming_vector(list_words=row['stemmed_content'], vocab_dict=dictionary_vocab), axis=1)
df_val["ham_vector"] = df_val.apply(lambda row : form_hamming_vector(list_words=row['stemmed_content'], vocab_dict=dictionary_vocab), axis=1)
df_test["ham_vector"] = df_test.apply(lambda row : form_hamming_vector(list_words=row['stemmed_content'], vocab_dict=dictionary_vocab), axis=1)

"""## Create eucledian vectors by representing how many times"""

def form_eucledian_vector(list_words, vocab_dict):
    vec = np.zeros(len(vocab_dict)) # +1 for unknown word.
    
    ## Form a small dictionary to store each word count
    dict_local_vocab = {}
    for word in list_words:
        if word not in dict_local_vocab:
            dict_local_vocab[word] = 1
        else:
            dict_local_vocab[word] = dict_local_vocab[word] + 1

    # print(dict_local_vocab)

    # unknown_word_count = 1
    for word in dict_local_vocab:
        if word not in vocab_dict: ## add 1 to unkown word index.
            continue
            # vec[UNKNOWN_WORD_INDEX] = unknown_word_count
            # unknown_word_count += 1
        else: ## word is present in vocab, get index.
            (value, idx) = vocab_dict[word] 
            vec[idx] = dict_local_vocab[word] ## replace with value of this word instead of 1.

    del dict_local_vocab
    return vec

df_train["euc_vector"] = df_train.apply(lambda row : form_eucledian_vector(list_words=row['stemmed_content'], vocab_dict=dictionary_vocab), axis=1)
df_val["euc_vector"] = df_val.apply(lambda row : form_eucledian_vector(list_words=row['stemmed_content'], vocab_dict=dictionary_vocab), axis=1)
df_test["euc_vector"] = df_test.apply(lambda row : form_eucledian_vector(list_words=row['stemmed_content'], vocab_dict=dictionary_vocab), axis=1)

# FOLDER_DATASET = '/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Dataframes-Dataset'
# df_train.to_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Dataframes-Dataset/train.csv', index=False)
# df_val.to_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Dataframes-Dataset/val.csv', index=False)
# df_test.to_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Dataframes-Dataset/test.csv', index=False)

"""# Stack each of these vertically to form hamming and eucledian vectors."""

def get_2D_vector(list_np_arr):
    vec_2D = np.zeros((len(list_np_arr), len(list_np_arr[0])))
    idx = 0
    for vec in list_np_arr:
        vec_2D[idx] = vec
        idx += 1
    return vec_2D

vocab_size = len(dictionary_vocab)
print(f"vocab_size = {vocab_size}")

num_documents = len(df_train)
print(f"num_documents = {num_documents}")

print(df_train.columns.values)

### Obtain vectors for train set.

hamming_vectors_2D = get_2D_vector(list_np_arr=df_train['ham_vector'].values)
eucledian_vectors_2D = get_2D_vector(list_np_arr=df_train['euc_vector'].values)
print(hamming_vectors_2D.shape, eucledian_vectors_2D.shape)

"""## For now save these dataframes"""

# df_train.to_csv("/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/train.csv", index=False)
# df_val.to_csv("/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/val.csv", index=False)
# df_test.to_csv("/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/test.csv", index=False)

"""## Now, finally create TF-IDF and pickel dump everything."""

def compute_TF_IDF_all_train_set(hamming_vectors_2D, eucledian_vectors_2D, alpha=0.0001, beta=0.0001):
    num_documents, num_words = eucledian_vectors_2D.shape
    print(f"num_documents, num_words = {num_documents, num_words}")

    TF = np.zeros((num_documents, num_words))

    IDF = np.zeros((1, num_words))

    ## Calculate TF(d, w) = N(d, w)/W(d) ; where N(d, w): count(w) in document d , W(d): Total #words in document d
    for itr in range(num_documents): ## iterate row-wise
        doc_eucledian = eucledian_vectors_2D[itr]
        total_num_words_doc_eucledian = num_words - np.sum(doc_eucledian == 0)        
        TF[itr] = doc_eucledian/total_num_words_doc_eucledian

        # if itr == 1:
            # print(f"itr = {itr}, doc_euc[itr] = {np.unique(doc_eucledian[itr], return_counts=True)}")
            # print(f"itr = {itr}, TF[itr] = {np.unique(TF[itr], return_counts=True)}")
            # break

    ## Calculate IDF(d, w) = log( (D + alpha)/(C(w) + beta) ) ; C(w) -> Total # docs with word 'w' ; D -> Total # documents
    D = num_documents
    for itr in range(num_words): ## itereate col-wise
        C_w = np.sum(hamming_vectors_2D[:, itr] == 1) ## first calculate C(w)
        
        IDF[:, itr] = np.log( (D + alpha) / (C_w + beta) )

        # break
        
    TF_IDF = TF*IDF
    del TF
    # del IDF
    return TF_IDF, IDF

TF_IDF_whole_corpus, IDF_whole_corpus = compute_TF_IDF_all_train_set(hamming_vectors_2D=hamming_vectors_2D, eucledian_vectors_2D=eucledian_vectors_2D)
print(f"TF_IDF_whole_corpus.shape = {TF_IDF_whole_corpus.shape}")
print(f"IDF_whole_corpus.shape = {IDF_whole_corpus.shape}")

def form_TF_IDF_for_val_test(list_words, IDF, hamming_vectors_2D, eucledian_vectors_2D, vocab_dict, alpha=0.0001, beta=0.0001, epslion=0.000001):
    num_docs_train, num_words = eucledian_vectors_2D.shape

    TF = np.zeros((1, num_words))
    
    euc_vec = form_eucledian_vector(list_words=list_words, vocab_dict=vocab_dict)
    
    W_d_num_words_in_document = 0
    for word in list_words:
        if word in vocab_dict:
            W_d_num_words_in_document += 1

    TF = euc_vec/(W_d_num_words_in_document + epslion)

    TF_IDF = TF*IDF

    return TF_IDF

words_val = df_val['stemmed_content'].iloc[0]

TF_IDF = form_TF_IDF_for_val_test(list_words=words_val, IDF=IDF_whole_corpus, hamming_vectors_2D=hamming_vectors_2D, 
                    eucledian_vectors_2D=eucledian_vectors_2D, vocab_dict=dictionary_vocab, alpha=0.0001, beta=0.0001)
print(f"TF_IDF.shape = {TF_IDF.shape}")
# print(np.unique(TF_IDF, return_counts=True))

"""# Now we start with K-Nearest Neighbor

#### Form validation 2D set
"""

hamming_vectors_2D_validation = get_2D_vector(list_np_arr=df_val['ham_vector'].values)
euclidean_vectors_2D_validation = get_2D_vector(list_np_arr=df_val['euc_vector'].values)
print(hamming_vectors_2D_validation.shape, euclidean_vectors_2D_validation.shape)

tf_idf_validation_set = np.asarray([
    form_TF_IDF_for_val_test(list_words=x, IDF=IDF_whole_corpus, hamming_vectors_2D=hamming_vectors_2D, 
        eucledian_vectors_2D=eucledian_vectors_2D, vocab_dict=dictionary_vocab, alpha=0.0001, beta=0.0001) for x in df_val['stemmed_content'].values
])
tf_idf_validation_set = tf_idf_validation_set.reshape(tf_idf_validation_set.shape[0], -1)
print(f"tf_idf_validation_set.shape = {tf_idf_validation_set.shape}")

"""## Similarity functions for 2D vectorized"""

###### For vectorized ############
def ham(a, b):
    # return np.count_nonzero((a!=b[:, None]), axis=-1) ## returns 3-D matrix. Don't use.
    return pairwise_distances(a, b, metric='euclidean') ## since binary vectors will return hamming-distance

def euclidean(a, b):
    # return np.linalg.norm((a-b[:, None]), axis=-1)
    return pairwise_distances(a, b, metric='euclidean')


def cosine_similarity(a, b):
    # return (b.dot(a.T))/(np.linalg.norm(a, axis=1) * np.linalg.norm(b[:, None], axis=-1)) ## (a@b.T).T  ## to get r2*r1
    return pairwise_distances(a, b, metric='cosine')

#@title Similarity functions without pairwise-distances
####### For single loop ############
# ## Similarity functions.
# def ham(a, b):
#     return np.count_nonzero((a!=b), axis=1)

# def euclidean(a, b):
#     return np.linalg.norm((a-b), axis=1)

# ### https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
# def cosine_similarity(a, b):
#     cos_sim = np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#     return cos_sim


# a = np.array([x for x in range(40)])
# a = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
# a = a.reshape(5, -1)
# print("a: \n", a)

# b = np.array([1, 0, 1])
# b = b.reshape(1, -1)
# print("\nb: ", b)


# # print(ham(a, b))
# print(f"\nEuclidean(a, b) = {euclidean(a, b)}")

# a = np.array([2, 1, 3, 4, 5, 1, 10, 3, 22])
# n = 4
# indices_top = (-a).argsort()[:n]
# print(f"\n n = {n}, indices_top = {indices_top}")
# print(a[indices_top])

labels_train = df_train['Label'].values
print(labels_train.shape)

labels_val = df_val['Label'].values
print(labels_val.shape)

val_euc_0 = euclidean_vectors_2D_validation[0:2]
print(val_euc_0.shape)

### KNN Vectorized.
class KNN:
    def __init__(self, mode="hamming", to_print=False):
        self.mode = mode
        if to_print == True:
            print(f"KNN __init__(mode={mode})")
        

    def compute_distances(self, v):
        # print(f"v.shape = {v.shape}")
        if self.mode == "hamming":
            self.distances = ham(v, self.vectors_corpus)
        elif self.mode == "euclidean":
            self.distances = euclidean(v, self.vectors_corpus)
        elif self.mode == 'cosine_similarity':
            self.distances = cosine_similarity(v, self.vectors_corpus)

        # print(f"After compute_distances mode={self.mode}, distances.shape = {self.distances.shape}")
        # print(f"{self.distances}")

    def populate_vectors(self, vectors_corpus, labels):
        self.vectors_corpus = vectors_corpus
        self.labels = labels

    def predict(self, v, num_neighbors, compute_distance_flag=True):
        K = num_neighbors
        ## compute distances by using suitable similarity function.
        if compute_distance_flag == True:
            self.compute_distances(v)

        ## take argmax top results indices
        ## (-v.T).argsort(axis=0)[:K].reshape(-1, )
        indices_top = (self.distances.T).argsort(axis=0)[:K] ## highest value is negative distance ? Don't know why, +ve should be taken.

        ## apply indices to labels
        top_labels = self.labels[indices_top]

        ## take majority vote and return the label
        top_most_label = stats.mode(top_labels)[0][0]

        ## return the max label
        return top_most_label

# ################################################# Test #################################################

"""## Applying tests on validation set for KNN"""

# %%time
## Initialize results dataframe
column_names = ["Similarity-Measure", "K", "Accuracy(%)"]
df_results_knn = pd.DataFrame(columns=column_names)

### Initialize objects
knn_ham = KNN(mode='hamming')
knn_euc = KNN(mode='euclidean')
knn_cosine = KNN(mode='cosine_similarity')

### Populate initial vectors
knn_ham.populate_vectors(hamming_vectors_2D, labels_train)
knn_euc.populate_vectors(eucledian_vectors_2D, labels_train)
knn_cosine.populate_vectors(TF_IDF_whole_corpus, labels_train)

### Compute distances for each val-set
knn_ham.compute_distances(hamming_vectors_2D_validation)
knn_euc.compute_distances(euclidean_vectors_2D_validation)
knn_cosine.compute_distances(tf_idf_validation_set)

### Predict for each value of K
for k in [1, 3, 5]: ## [1, 3, 5, 7, 9, 11, 13, 15]:
    print(f"Predicting for k = {k}")
    ## Predict and Append to dataframe.  
    ## Signature: def predict(self, v, num_neighbors, compute_distance_flag=True)
    y_preds = knn_ham.predict(hamming_vectors_2D_validation, num_neighbors=k, compute_distance_flag=False)
    acc = np.sum(labels_val==y_preds)/len(labels_val)*100 ## accuracy_score(y_true=labels_val, y_pred=y_preds, normalize=True)*100
    df_results_knn = add_to_dataframe(df_old=df_results_knn, to_add=["Hamming", k, acc])
    
    y_preds = knn_euc.predict(euclidean_vectors_2D_validation, num_neighbors=k, compute_distance_flag=False)
    acc = accuracy_score(y_true=labels_val, y_pred=y_preds, normalize=True)*100
    df_results_knn = add_to_dataframe(df_old=df_results_knn, to_add=["Euclidean", k, acc])

    y_preds = knn_cosine.predict(tf_idf_validation_set, num_neighbors=k, compute_distance_flag=False)
    acc = accuracy_score(y_true=labels_val, y_pred=y_preds, normalize=True)*100
    df_results_knn = add_to_dataframe(df_old=df_results_knn, to_add=["TF-IDF-cosine-sim", k, acc])
    

### Delete each objects
del knn_ham
del knn_euc
del knn_cosine

### Print
print(f"len df_results_knn = {len(df_results_knn)}")
print(df_results_knn.head(3))

print(f"len df_results_knn = {len(df_results_knn)}")
print(df_results_knn)
# df_results_knn.to_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Using-2D-Pairwise/KNN-val-topics.csv', index=False)

# mat = df_results_knn.values
# print(mat.shape)
# print(mat)

"""## Naive Bayes

### Combine all documents of each class i into one document
"""

# unique_labels = np.unique(df_train['Label'].values)
# dictionary_list_words_for_NB = {}
# # for label in unique_labels:
# #     if label not in dictionary_list_words:
# #         dictionary_list_words_for_NB[label] = [] ## create new list.
    
# for (list_words, label) in zip(df_train['stemmed_content'].values, df_train['Label'].values):
#     if label not in dictionary_list_words_for_NB:
#         dictionary_list_words_for_NB[label] = [] ## create new list.
#     dictionary_list_words_for_NB[label].append(list_words)

# ## reduce/flat-out to make 1D list.
# ## https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
# for label in dictionary_list_words_for_NB:
#     dictionary_list_words_for_NB[label] = reduce(operator.concat, dictionary_list_words_for_NB[label])

## https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

class NaiveBayes:
    def __init__(self, vocab_size, alpha=0.01, to_print=False):
        if to_print == True:
            print(f"NaiveBayes __init(alpha={alpha})__")
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.dictionary_list_words_for_NB = {}
        self.dictionary_prior_probabilities = {}
        self.dictionary_count_words_per_class = {}
        self.dictionary_total_words_per_class = {}
    

    def fit(self, list_list_words, labels):
        self.list_list_words = list_list_words
        self.labels = labels
        self.form_dictionary_list_words()
        self.compute_probabilities()


    def form_dictionary_list_words(self):
        for (list_words, label) in zip(self.list_list_words, self.labels):
            if label not in self.dictionary_list_words_for_NB:
                self.dictionary_list_words_for_NB[label] = [] ## create new list.
            self.dictionary_list_words_for_NB[label].append(list_words)
        
        for label in self.dictionary_list_words_for_NB: ## reduce/flat-out to make 1D list.
            self.dictionary_list_words_for_NB[label] = reduce(operator.concat, self.dictionary_list_words_for_NB[label])


    def compute_probabilities(self):
        ## Compute prior probabilities/initial guesses
        (classes, cnts) = np.unique(self.labels, return_counts=True)
        cnts = cnts/np.sum(cnts) ## C_i / (C_1 + C_2 + ... + C_n)
        for (lab, itr) in zip(classes, range(len(classes))):
            self.dictionary_prior_probabilities[lab] = cnts[itr]

        ## Compute per-word probabilities

        ## Counter increment
        for lab in classes:
            num_words_this_class = 0
            self.dictionary_count_words_per_class[lab] = {}
            for word in self.dictionary_list_words_for_NB[lab]:
                if word not in self.dictionary_count_words_per_class[lab]:
                    self.dictionary_count_words_per_class[lab][word] = 0 ## initialize counter to 0.
                self.dictionary_count_words_per_class[lab][word] = self.dictionary_count_words_per_class[lab][word] + 1 ## increment counter
                num_words_this_class += 1
            self.dictionary_total_words_per_class[lab] = num_words_this_class

    def predict(self, list_words):
        ## Compute probabilities for each label.
        dict_probabilities_per_class = {}

        for lab in np.unique(self.labels):
            prob_log_curr_class = np.log(self.dictionary_prior_probabilities[lab]) ## start with prior probabilities
            # prob_log_curr_class = (self.dictionary_prior_probabilities[lab]) ## start with prior probabilities
            # print(f"Before, prob_log_curr_class = {prob_log_curr_class}")
            for word in list_words: ## iterate per word
                if word in self.dictionary_count_words_per_class[lab]: ## if word exists in THIS document.
                    ## use smoothing factor alpha
                    # prob_log_curr_class += np.log((self.dictionary_count_words_per_class[lab][word] + self.alpha)/(self.dictionary_total_words_per_class[lab] + self.alpha*self.vocab_size)) 
                    word_prob = self.dictionary_count_words_per_class[lab][word]
                    # prob_log_curr_class = prob_log_curr_class*word_prob
                else:
                    word_prob = 0
                prob_log_curr_class += np.log( (word_prob + self.alpha)/(self.dictionary_total_words_per_class[lab] + self.alpha*self.vocab_size) )


            dict_probabilities_per_class[lab] = prob_log_curr_class
            # print(dict_probabilities_per_class)

        ## Get max probability class.
        ## https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        return keywithmaxval(d=dict_probabilities_per_class)




######################################### Checking ############################################################
NB = NaiveBayes(alpha=0.01, vocab_size=len(dictionary_vocab))
NB.fit(list_list_words=df_train['stemmed_content'].values, labels=df_train['Label'].values)

check = df_val['stemmed_content'].iloc[0]
p = NB.predict(check)
print(p)

list_new = [
['coffee', 'tea', 'dew', 'dew', 'dew', 'dew'],
['coffee', 'noir', 'homelander', 'dew'],
['noir', 'noir', 'fool', 'fool', 'noir']
]

labs_new = [
    'bev',
    'supe',
    'misc'
]

vocab_size = 6

NB = NaiveBayes(alpha=0.01, vocab_size=6)
NB.fit(list_new, labs_new)
print(NB.predict(['coffee', 'tea', 'dew', 'dew', 'dew', 'dew']))

"""## Validation on NaiveBayes"""

column_names = ["ALPHA", "Accuracy(%)"]
df_results_NB = pd.DataFrame(columns=column_names)

## alpha_values = np.linspace(start=0.01, stop=1.0, num=100)
alpha_values = [0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.5, 0.75, 0.85, 1.0]
print(alpha_values)

### Validation NaiveBayes ###
for alpha in (alpha_values):
    NB = NaiveBayes(alpha=alpha, vocab_size=len(dictionary_vocab), to_print=True)
    NB.fit(list_list_words=df_train['stemmed_content'].values, labels=df_train['Label'].values)

    nb_acc = 0
    ## Predict each val set ##
    for (x, y) in (zip(df_val['stemmed_content'].values, df_val['Label'].values)):
        y_pred = NB.predict(x)
        if y_pred == y:
            nb_acc += 1


    ## Append to dataframe and print.
    # print(f"NB alpha = {alpha}, accuracy = {nb_acc/len(df_val)*100} %")
    df_results_NB = add_to_dataframe(df_old=df_results_NB, to_add=[alpha, nb_acc/len(df_val)*100])
    del NB

df_results_NB.sort_values(by=['Accuracy(%)'], inplace=True, ascending=False) ## SORT
print(df_results_NB.head(10))
# df_results_NB.to_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Using-2D-Pairwise/NB-val-topics.csv', index=False)

"""# Hypothesis testing on Test Dataset"""

del hamming_vectors_2D_validation
del euclidean_vectors_2D_validation
del tf_idf_validation_set

"""### Create and fit best performing models on validation set"""

words_val = df_val['stemmed_content'].iloc[0]
print(words_val)

##### KNN was K=5, TF-IDF #####
K_best = 5 ## 15
knn = KNN(mode='cosine_similarity', to_print=True)
knn.populate_vectors(TF_IDF_whole_corpus, labels_train)

# val_tf_idf = form_TF_IDF_for_val_test(list_words=words_val, IDF=IDF_whole_corpus, hamming_vectors_2D=hamming_vectors_2D, 
#                 eucledian_vectors_2D=eucledian_vectors_2D, vocab_dict=dictionary_vocab, alpha=0.0001, beta=0.0001)
# print(knn.predict(val_tf_idf, num_neighbors=5))

##### Naive Bayes was alpha = 0.02/0.04, we will take 0.04 #####
NB = NaiveBayes(alpha=0.04, vocab_size=len(dictionary_vocab), to_print=True)
NB.fit(list_list_words=df_train['stemmed_content'].values, labels=df_train['Label'].values)

print(NB.predict(words_val))

del df_train
del df_val

"""### Split test dataset 50 iterations per 10 of each topic"""

print(df_test.columns.values)

df_test.sort_values(by=['Label'], inplace=True)
df_test.drop(labels=['content', 'ham_vector', 'euc_vector'], axis=1, inplace=True)

column_names = ["KNN-Acc(%)", "NB-Acc(%)"]

df_results_test_set = pd.DataFrame(columns=column_names)

"""## Faster prediction of test set"""

print(f"K_best = {K_best}")

### For KNN ###
tf_idf_test_set = np.asarray([
    form_TF_IDF_for_val_test(list_words=x, IDF=IDF_whole_corpus, hamming_vectors_2D=hamming_vectors_2D, 
        eucledian_vectors_2D=eucledian_vectors_2D, vocab_dict=dictionary_vocab, alpha=0.0001, beta=0.0001) for x in df_test['stemmed_content'].values
])
tf_idf_test_set = tf_idf_test_set.reshape(tf_idf_test_set.shape[0], -1)
print(f"tf_idf_test_set.shape = {tf_idf_test_set.shape}")

y_preds_knn = knn.predict(tf_idf_test_set, num_neighbors=K_best)
print(f"y_preds_knn.shape = {y_preds_knn.shape}")

del tf_idf_test_set
del knn

### For Naive Bayes ###
y_preds_nb = []
for x in df_test['stemmed_content'].values:
    y_preds_nb.append( NB.predict(x) )
y_preds_nb = np.array(y_preds_nb)
print(f"y_preds_nb.shape = {y_preds_nb.shape}")
del NB

y_test = df_test['Label'].values

"""### Splitting now"""

num_docs_in_each_topic = 500 ##
initial_offset = np.array([i*num_docs_in_each_topic for i in range(0, 11)])
for counter_test in range(0, 50): ## run iterations 50 times
    offsets = initial_offset + counter_test*10
    start_indices = offsets
    end_indices = offsets + 10
    
    # print("\nCounter = ", counter_test)
    # print(offsets)
    # print(start_indices)
    # print(end_indices)

    ### Testing here ###
    nb_correct = knn_correct = 0
    for i in range(len(start_indices)): ## add all to list.
        ## print(np.unique(df_test.iloc[start_indices[i]:end_indices[i]]['Label'].values, return_counts=True), end=' ')
        ## for (x, y) in zip(df_test.iloc[start_indices[i]:end_indices[i]]['stemmed_content'].values, df_test.iloc[start_indices[i]:end_indices[i]]['Label'].values):
        for y, y_knn, y_nb in zip(y_test[start_indices[i]:end_indices[i]], y_preds_knn[start_indices[i]:end_indices[i]], y_preds_nb[start_indices[i]:end_indices[i]]):
            if y_nb == y:
                nb_correct += 1
            if y_knn == y:
                knn_correct += 1

    NUM_DOCUMENTS = 10* len(np.unique(y_test))
    knn_acc = knn_correct/( NUM_DOCUMENTS )*100  ## 10*11 total 110 test documents per iteration. [50 iterations]
    nb_acc = nb_correct/( NUM_DOCUMENTS )*100
    # print(f"KNN-Acc = {knn_acc}%, NB-Acc = {nb_acc}%")

    df_results_test_set = add_to_dataframe(df_old=df_results_test_set, to_add=[knn_acc, nb_acc])

# display(df_results_test_set)
print(df_results_test_set.head(5))

"""## Predictions using normal for each example"""

# num_docs_in_each_topic = 500 ##
# initial_offset = np.array([i*num_docs_in_each_topic for i in range(0, 11)])
# for counter_test in range(0, 50): ## run iterations 50 times
#     offsets = initial_offset + counter_test*10
#     start_indices = offsets
#     end_indices = offsets + 10
    
#     print("\nCounter = ", counter_test)
#     # print(offsets)
#     # print(start_indices)
#     # print(end_indices)

#     ### Testing here ###
#     nb_correct = knn_correct = 0
#     for i in range(len(start_indices)): ## add all to list.
#         # print(np.unique(df_test.iloc[start_indices[i]:end_indices[i]]['Label'].values, return_counts=True), end=' ')
#         for (x, y) in zip(df_test.iloc[start_indices[i]:end_indices[i]]['stemmed_content'].values, df_test.iloc[start_indices[i]:end_indices[i]]['Label'].values):
            
#             #### For KNN ####
#             x_tf_idf = form_TF_IDF_for_val_test(list_words=x, IDF=IDF_whole_corpus, hamming_vectors_2D=hamming_vectors_2D, 
#                 eucledian_vectors_2D=eucledian_vectors_2D, vocab_dict=dictionary_vocab, alpha=0.0001, beta=0.0001)
#             y_pred = knn.predict(x_tf_idf, num_neighbors=K_best)
#             if y_pred == y:
#                 knn_correct += 1
            
#             #### For Naive Bayes #####
#             y_pred = NB.predict(x)
#             if y_pred == y:
#                 nb_correct += 1
            
    
#     print(f"Done for counter_test = {counter_test}")

#     NUM_DOCUMENTS = 10* len(np.unique(df_test['Label'].values))
#     knn_acc = knn_correct/( NUM_DOCUMENTS )*100  ## 10*11 total 110 test documents per iteration. [50 iterations]
#     nb_acc = nb_correct/( NUM_DOCUMENTS )*100
#     print(f"KNN-Acc = {knn_acc}%, NB-Acc = {nb_acc}%")

#     df_results_test_set = add_to_dataframe(df_old=df_results_test_set, to_add=[knn_acc, nb_acc])

# df_results_test_set.to_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Using-2D-Pairwise/Test-Set-KNN-K=5-NB.csv', index=False)

print(f"len df_results_test_set = {len(df_results_test_set)}")

"""## Load and analyze."""

# df_results_test_set = pd.read_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Using-2D-Pairwise/Test-Set-KNN-K=5-NB.csv')
# print(df_results_test_set.head(5))

knn_acc = df_results_test_set['KNN-Acc(%)'].values
nb_acc = df_results_test_set['NB-Acc(%)'].values
print(knn_acc.shape, nb_acc.shape)

column_names = ["Method", "Mean-Acc(%)", "Std-dev", "Std-Error", "Minimum-Acc(%)", "Maximum-Acc(%)"]

df_stats = pd.DataFrame(columns=column_names)

## Summarized results.
df_stats = add_to_dataframe(df_old=df_stats, to_add=["KNN K=5, TF-IDF", np.mean(knn_acc), np.std(knn_acc), stats.sem(knn_acc, axis=None, ddof=0), min(knn_acc), max(knn_acc)])
df_stats = add_to_dataframe(df_old=df_stats, to_add=["NB alpha = 0.04", np.mean(nb_acc), np.std(nb_acc), stats.sem(nb_acc, axis=None, ddof=0), min(nb_acc), max(nb_acc)])

display(df_stats)

# df_stats.to_csv('/content/drive/My Drive/ML-Undergrad-Assignments-Projects/Assignment-2 Text NaiveBayes KNN/Using-2D-Pairwise/Summarized-Test-Set-KNN-K=5-NB.csv', index=False)

"""## Computing T-statistics"""

knn_acc = df_results_test_set['KNN-Acc(%)'].values
nb_acc = df_results_test_set['NB-Acc(%)'].values

## Using K = 5

## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

# ans = stats.ttest_rel(nb_acc, knn_acc)
# ans = stats.ttest_ind(nb_acc, knn_acc, equal_var=True)
ans = stats.ttest_rel(nb_acc, knn_acc)
print(ans)

