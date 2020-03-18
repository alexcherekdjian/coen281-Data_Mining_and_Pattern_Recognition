import numpy as np
import scipy as sp
import string
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


# defined functions
def filterLen(docs, minlen):
    r""" filter out terms that are too short. 
    docs is a list of lists, each inner list is a document represented as a list of words
    minlen is the minimum length of the word to keep
    """
    return [ [t for t in d if len(t) >= minlen ] for d in docs ]

# (how many times each word is found in the collection)
def plotWf(docs, plot=True, logscale=True):
    r"""Get collection-wide word frequencies and optionally plot them."""
    words = defaultdict(int)
    for d in docs:
        for w in d:
            words[w] += 1
            
    return words

# (how many documents each word is found in)
def plotDf(docs, plot=True, logscale=True):
    r"""Get collection-wide word frequencies and optionally plot them."""
    # document word frequency
    df = defaultdict(int)
    for d in docs:
        for w in set(d):
            df[w] += 1

    return df

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )

# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

def Punctuation(string): 
  
    # punctuation marks 
    punctuations = '''!()-[]{};0123456789:'"\,<>./?@#$%^&*_~'''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

    string = string.lower()
    return string

###################### MAIN DRIVER ######################

# open docs file and read its lines
with open("train.dat", "r") as fh:
    lines_train = fh.readlines()  

with open("test.dat", "r") as fh:
    lines_test = fh.readlines()  

# split up words by whitespace
docs_train = [l.split() for l in lines_train]
docs_test = [l.split() for l in lines_test]

# get training answers 
y_train = []
for document in docs_train:
    y_train.append(document.pop(0))

# append test to train docs
docs = docs_train + docs_test

# get stop words
stop_words = set(stopwords.words('english')) 

# filter out stop words in documents
docs_filtered = []

for sample in docs:
    filtered_sentence = []
    
    for w in sample:
        if w not in stop_words:
            filtered_sentence.append(w)

    docs_filtered.append(filtered_sentence)

# filter out punctuation in documents
punctuation_filtered = []

for sample in docs_filtered:
    filtered_sentence = []
    for w in sample:
        result = Punctuation(w)
        filtered_sentence.append(result)

    punctuation_filtered.append(filtered_sentence)

# stem words using porter algorithm
stemmer= PorterStemmer()
porter_filtered = []

printProgressBar(0, len(punctuation_filtered), prefix = 'Progress:', suffix = 'Complete', length = 50)

i = 0
for sample in punctuation_filtered:
    filtered_sentence = []
    for w in sample:
        result = stemmer.stem(w)
        filtered_sentence.append(result)

    porter_filtered.append(filtered_sentence)
    i = i+1
    printProgressBar(i, len(punctuation_filtered), prefix = 'Progress:', suffix = 'Complete', length = 50)

# filter documents by word length 
docs_final = filterLen(porter_filtered, 4)

print("filtering complete. Now predicting!")

# build the matrix
docs_mat = build_matrix(docs_final)
csr_info(docs_mat)

# split back into train and test
x_train = docs_mat[:14438]
x_test = docs_mat[14438:]

# normalize via idf
mat_train = csr_idf(x_train, copy=True)
mat_test = csr_idf(x_test, copy=True)

# l2 normalize
sparse_mat_train = csr_l2normalize(mat_train, copy=True)
sparse_mat_test = csr_l2normalize(mat_test, copy=True)

print(sparse_mat_train.shape)
print(sparse_mat_test.shape)

# k value and y predictions list
k = 111
y_pred = []

printProgressBar(0, x_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
progress = 0

# dot product between vector and matrix
for i in range(0, x_test.shape[0]):
    sample_dp = []
    # calculate dot product
    dp3 = sparse_mat_train.dot(sparse_mat_test[i].T)
    dp3 = dp3.reshape(1, 14438).todense()

    # get max k values from array
    indexes = np.argpartition(dp3,-k)[-k:].tolist()
    indexes = indexes[0][-k:]
    
    # get cosine sim values
    max_cos_values = np.partition(dp3, -k)[-k:].tolist()
    max_cos_values = max_cos_values[0][-k:]
    
    # get prediction values
    predicts = []
    for index in indexes:
        predicts.append(y_train[index])
    
    # init vote dictionary
    votes_dict = {} 
    votes_dict[1] = 0.0
    votes_dict[2] = 0.0
    votes_dict[3] = 0.0
    votes_dict[4] = 0.0
    votes_dict[5] = 0.0
    
    # ensure predictions are not strings
    predicts = [int(i) for i in predicts]
    
    # add up votes
    for y_predicted, cos_val in zip(predicts, max_cos_values):
        votes_dict[y_predicted] += cos_val
    
    # get max prediction
    key_max = max(votes_dict.keys(), key=(lambda k: votes_dict[k]))
    
    # append answer
    y_pred.append(key_max)
    progress+=1
    printProgressBar(progress, x_test.shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)

# write to a file
print("writing file . . .")
printProgressBar(0, len(y_pred), prefix = 'Progress:', suffix = 'Complete', length = 50)

f = open("pred_111.txt", "w")
for i in range (0, len(y_pred)):
    f.write(str(y_pred[i]) + "\n")
    printProgressBar(i+1, len(y_pred), prefix = 'Progress:', suffix = 'Complete', length = 50)

f.close()
