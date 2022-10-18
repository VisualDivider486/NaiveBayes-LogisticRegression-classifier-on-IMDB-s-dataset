import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=1000,skip_top=50)

word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])

vocabulary = list()
for text in x_train:
  tokens = text.split()
  vocabulary.extend(tokens)

vocabulary = set(vocabulary)
print(len(vocabulary))

from tqdm import tqdm

x_train_binary = list()
x_test_binary = list()

for text in range(500):
  tokens = tqdm(x_train[text].split())
  binary_vector = list()
  for vocab_token in vocabulary:
    if vocab_token in tokens:
      binary_vector.append(1)
    else:
      binary_vector.append(0)
  x_train_binary.append(binary_vector)

x_train_binary = np.array(x_train_binary)

for text in range(500):
  tokens = tqdm(x_test[text].split())
  binary_vector = list()
  for vocab_token in vocabulary:
    if vocab_token in tokens:
      binary_vector.append(1)
    else:
      binary_vector.append(0)
  x_test_binary.append(binary_vector)

x_test_binary = np.array(x_test_binary)

y_train_x=np.ones(500)
for i in range(500):
  y_train_x[i]=y_train[i]

y_test_x=np.ones(500)
for i in range(500):
  y_test_x[i]=y_test[i]

  from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x_train_binary,y_train_x,test_size=0.20)

import numpy as np
  
def accuracy(y_test,predictions):
    acc=0
    for x in range(y_test.size):
      if(y_test[x]==predictions[x]):
        acc +=1
    acc =acc/len(y_test)
    return acc

def positive_precision(y_test,predictions):
    positive_pres=0
    for x in range(y_test.size):
      if(y_test[x]==predictions[x] and predictions[x]==1 ):
        positive_pres +=1
    positive_pres =positive_pres/np.count_nonzero(predictions)
    return positive_pres

def negative_precision(y_test,predictions):
    negative_pres=0
    for x in range(y_test.size):
      if(y_test[x]==predictions[x] and predictions[x]==0 ):
        negative_pres +=1
    negative_pres =negative_pres/(len(predictions)-np.count_nonzero(predictions))
    return negative_pres

def positive_recall(y_test,predictions):
    positive_rec=0
    for x in range(y_test.size):
      if(y_test[x]==predictions[x] and predictions[x]==1 ):
        positive_rec +=1
    positive_rec =positive_rec/np.count_nonzero(y_test)
    return positive_rec

def negative_recall(y_test,predictions):
    negative_rec=0
    for x in range(y_test.size):
      if(y_test[x]==predictions[x] and predictions[x]==0 ):
        negative_rec +=1
    negative_rec =negative_rec/(len(predictions)-np.count_nonzero(y_test))
    return negative_rec

def macro_recall(recall1,recall2):
    return (recall1+recall2)/2
def macro_precision(precision1,precision2):
    return (precision1+precision2)/2

def F1(recall,precision):
    return (2*precision*recall)/(precision+recall)


def classification_report(y_test,predictions):
    print("Accuracy is: ",accuracy(y_test,predictions)," %")
    print("Positive precision is: ",positive_precision(y_test,predictions)," %")
    print("Negative precision is: ",negative_precision(y_test,predictions)," %")
    print("Positive recall is: ",positive_recall(y_test,predictions)," %")
    print("Negative recall is: ",negative_recall(y_test,predictions)," %")
    print("F1 is: ",F1(macro_recall(positive_recall(y_test,predictions),negative_recall(y_test,predictions)),macro_precision(positive_precision(y_test,predictions),negative_precision(y_test,predictions)))," %")
    print("\n")

import numpy as np

class LogisticRegression():

  def sigmoid(self,z):
        return 1/(1+np.exp(-z))

  def __init__ (self,learning_rate, n_iterations):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
      
  def fit(self,x,y):
        examples= x.shape[0]
        feats = x.shape[1]
        
        self.weights=np.zeros(feats)
        self.bias=0
        
        for _ in range(self.n_iterations):
            
            result=self.sigmoid(np.dot(x,self.weights)+self.bias)

            dw = (1/examples)*(np.dot(x.T, (result-y.T).T))
            db = (1/examples)*(np.sum(result-y.T))
            
            self.weights = self.weights - (self.learning_rate * (dw.T))
            self.bias = self.bias - (self.learning_rate * db)
       
    
  def predict(self, x):
        prediction = np.zeros(x.shape[0])
        result=self.sigmoid(np.dot(x,self.weights)+self.bias)
        for x in range(result.shape[0]):
          if result[x] > 0.5:
            prediction[x] = 1
        return prediction

      
print("Using Logistic Regression")
print("\n")
predictionLR=LogisticRegression(0.05,1000)
predictionLR.fit(x_train,y_train)
predictionLR=predictionLR.predict(x_test)
predictionLR=np.array(predictionLR)
classification_report(y_test,predictionLR)
