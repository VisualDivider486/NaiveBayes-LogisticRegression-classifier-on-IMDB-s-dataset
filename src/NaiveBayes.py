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

class NaiveBayes():

  #Ypologizei thn pithanothta kathe kathgorias, dhladh thn pithanothta na einai kalh kai kakh kritikh/P(C=1),P(C=0)
  def class_probability(self,y):
    positive = 0
    for x in range(y.size):
        if(y_train[x] !=0 ):
          positive+=1
      
    positive_prob = (positive+1)/ (y.size+2)#egine laplace
    negative_prob = 1-positive_prob
    self.class_prob = np.array([positive_prob,negative_prob])

  #Ypologizei thn desmevmenh pithanothta kathe idiothtas      
  def conditional_probability(self,x,y):
        examples= x.shape[0]
        feats = x.shape[1]
        positives = 0
        for i in range(y.size): #ypologizei posa arnytika kai thetika exei to y
          if(y[i] !=0 ):
            positives+=1

        negatives = examples-positives

        #arxikopoihsh me assous 2 array poy tha periexoyn tis pithanothtes 
        self.conditional_prob_0=np.ones(feats)
        self.conditional_prob_1=np.ones(feats)

        for feat in range(feats): #metraei poses fores mia sugkekrimenh leksh
    # einai se thetiko h arnhtiko review. p.x h leksh boring emfanizete 30 fores se thetiko review kai 120 se arnhtiko kai
    #meta vriskoume thn pithanothta tous na emfanistoyn 

          column=[row[feat] for row in x]

          count_pos=0 
          count_neg=0
          for x_i,y_i in zip(column,y_train):
            if x_i==1 and y_i!=0:
              count_pos=count_pos+1
            elif x_i==1 and y_i==0:
              count_neg=count_neg+1
            self.conditional_prob_0[feat]=count_pos/positives #ypologismos pithanotitas kai eisagogh ston "thetiko" array
            self.conditional_prob_1[feat]=count_neg/negatives #ypologismos pithanotitas kai eisagogh ston "arnhtiko" array

          #merge array
          self.conditional_prob = np.array([self.conditional_prob_0,self.conditional_prob_1])
      
        return self.conditional_prob #epistrefei enan array me 2 grammes, h kathe mia exei thn desmevmenh pithanotha kathe idiothtas na einai thetikh 
                                                                                                            #kai arnhtikh antistoixa gia kathe grammh
    
  
  def predict(self,x):

        #.Positive-->Row(0) Negative-->Row(1)
        sums=np.ones((2,x.shape[0]))
        
        for numerator,x_i in enumerate(x):
            for numeratorx,feat in enumerate(x_i):
                if (feat==1):
                    sums[0][numerator]+=np.log(self.conditional_prob[0][numeratorx])
                    sums[1][numerator]+=np.log(self.conditional_prob[1][numeratorx])
                else:
                    sums[0][numerator]+=np.log((1-self.conditional_prob[0][numeratorx]))
                    sums[1][numerator]+=np.log((1-self.conditional_prob[1][numeratorx]))
        
        
        sums=np.exp(sums)
        sums[0]=sums[0]*self.class_prob[0] #Athroisma P(Xi=xi|C=0)*P(C=0)
        sums[1]=sums[1]*self.class_prob[1] #Athroisma P(Xi=xi|C=1)*P(C=1)

        #o prediction einai ena aplo array poy periexei mesa 0 h 1 analoga me to an mia idiothta einai se thetiko h arnhtiko review kai epilegete
        #kanontas sygkrish  ton grammon toy sums
        prediction=np.zeros(x.shape[0])
        i=0
        for x in range(x.shape[0]):
            if(sums[0][x]>sums[1][x]):
              prediction[x]=1 
            else:
              prediction[x]=0
        return prediction

print("Using Naive Bayes Algorithm")
print("\n")
predictionNB=NaiveBayes()
predictionNB.class_probability(y_train)
predictionNB.conditional_probability(x_train,y_train)
predictionNB=predictionNB.predict(x_test)
classification_report(y_test,predictionNB)