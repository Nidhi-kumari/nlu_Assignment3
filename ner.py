
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np



# In[2]:


i=0
lis=[]
with open("ner.txt","r+") as file:
    for line in file:
        if line.startswith('\n'):
            i+=1
            continue
        line1='sent'+str(i) + ' '+line
        #print(line1) 
        lis.append(line1)
with open("out.txt", "w") as f1:
    f1.writelines(lis)



# In[3]:


data = pd.read_csv("out.txt",sep=" ",encoding="latin1",names=["Sentence #","Word", "Tag"])


# In[4]:


words = list(set(data["Word"].values))
words.append("ENDPAD")


# In[5]:


n_words = len(words); n_words


# In[6]:


tags = list(set(data["Tag"].values))


# In[7]:


n_tags = len(tags); n_tags


# In[8]:


tags


# In[9]:


data.dtypes


# In[10]:


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w,t) for w,t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        #print(self.sentences)
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[11]:


getter = SentenceGetter(data)


# In[12]:


sentences = getter.sentences


# In[13]:


#sentences


# In[14]:


max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[15]:


from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]


# In[16]:


X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)


# In[17]:


#X


# In[18]:


y = [[tag2idx[w[1]] for w in s] for s in sentences]


# In[19]:


y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# In[20]:


from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)


# In[21]:


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


# In[22]:


from keras_contrib.layers import CRF


# In[23]:


input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output


# In[24]:


model = Model(input, out)


# In[25]:


model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])


# In[26]:


model.summary()


# In[30]:


history = model.fit(X_tr, np.array(y_tr), batch_size=1000, epochs=4,
                    validation_split=0.1, verbose=1)


# In[ ]:


misclassification=0
total=0
actual=[]
predicted=[]

for i in range(X_te.shape[0]):
    p = model.predict(np.array([X_te[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_te[i], -1)
    print("{:15}   {:5}    {}".format("Word", "Actual", "Predicted"))
    print(30 * "=")
    for w, t, pred in zip(X_te[i], true, p[0]):
        total+=1
        if w != 0:
            print("     {:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))
            actual.append(tags[t])
            predicted.append(tags[pred])
            if tags[t]!=tags[pred]:
                misclassification+=1
    print("\n\n")


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(actual, predicted)


# In[46]:


hist = pd.DataFrame(history.history)
import matplotlib.pyplot as plt
import pylab 
plt.style.use("ggplot")
plt.figure(figsize=(12,12))
pylab.plot(hist["acc"],label='training accuracy')
pylab.xlabel("no.of iteration")
pylab.ylabel("accuracy")
pylab.plot(hist["val_acc"],label='validation accuracy')
pylab.legend(loc='upper left')
plt.show()


# In[48]:



from sklearn.metrics import confusion_matrix
p=confusion_matrix(actual, predicted)


# In[49]:


print(p)


# In[57]:


import itertools
import matplotlib.pyplot as plt
plt.figure()
plot_confusion_matrix(p, classes=tags, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[61]:


plt.figure()
plot_confusion_matrix(p, classes=tags,
                      title='Confusion matrix, without normalization')
plt.show()


# In[52]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[64]:


from sklearn.metrics import classification_report

report = classification_report(y_pred=predicted, y_true=actual)


# In[65]:


print(report)

