import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import get_ipython

if get_ipython():
    print("jupyter envirionment")
    from tqdm import tqdm_notebook as tqdm
else:
    print("command shell envirionment")
    from tqdm import tqdm


class OneHotEncoder() :
  def __init__(self, docs) :
    # 고유 단어와 인덱스를 매칭시켜주는 사전 생성
    self.w2i = defaultdict(lambda:len(self.w2i))
    [self.w2i[w] for d in docs for w in d]
    self.i2w = {v:k for k,v in self.w2i.items()}

  def _get_one_hot_vector(self, w):
    v = [0]*len(self.w2i)
    v[self.w2i[w]] = 1
    return v

  def encode(self, docs) :
    ret = []
    for d in docs :
      tmp = []
      for w in d :
        tmp.append(self._get_one_hot_vector(w))
      ret.append(tmp)
    return ret

  def decode(self, v) :
    return self.i2w[v.index(1)]


class Word2Vec() :
  def _slide_window(self, encoded_docs, win_size = 2) :
    ret = []

    for d in encoded_docs : 
      win_doc = []
      for i, w in enumerate(d) :
        s, e = max(0, i - win_size), min(len(d), i + win_size)
        context = d[s:e+1]
        center = context.pop(i-s)
        win_doc.append((center, context))
      ret.append(win_doc)
    return ret

  def _softmax(self, output) :
    return np.exp(output) / np.sum(np.exp(output))

  #SGD to backpropagate errors 
  def _backpropagation(self, W1, W2, hidden, predict, center, context, learning_rate) :
    y_hat = self._softmax(predict)
    err = (y_hat - context).sum(axis=0)

    delta_W2 = np.outer(hidden, err)
    delta_W1 = np.outer(center, np.dot(W2, err))

    W1 = W1 - learning_rate * delta_W1
    W2 = W2 - learning_rate * delta_W2

    return W1, W2

  def embedding(self, docs, method = 'sg', win_size = 2, embedding_size = 5, learning_rate = 0.001, epoch = 10 ) :
    if method == 'sg' :
      return self._skipgram(docs, win_size, embedding_size, learning_rate, epoch)
    elif method == 'cbow' :
      return self._cbow(docs, win_size, embedding_size, learning_rate, epoch)

  def _cbow(self, docs, win_size = 2, embedding_size = 5, learning_rate = 0.001, epoch = 10) :
    tokenized_docs = [d.split() for d in docs]
    ohe = OneHotEncoder(tokenized_docs)
    encoded_docs = ohe.encode(tokenized_docs)
    
    onehot_size = len(encoded_docs[0][0])
    W1 = np.random.rand(onehot_size, embedding_size)
    W2 = np.random.rand(embedding_size, onehot_size)

    sliding_docs = self._slide_window(encoded_docs, win_size)
    for i in tqdm(range(epoch), desc='word embedding') :
      for d in  sliding_docs:
        for center, context in d : 
          #이 부분 고쳐햐암
          hidden = np.dot(context, W1)
          predict = np.dot(hidden, W2)
          W1, W2 = self._backpropagation(W1, W2, hidden, predict, context, center, learning_rate)
      #self.loss += C*np.log(np.sum(np.exp(self.u))) 
    
    self.i2w = ohe.i2w
    self.wv_ = {self.i2w[i]:list(we) for i, we in enumerate(W1)}
    self.word_vectors = W1

    return self.wv_

    pass

  def _skipgram(self, docs, win_size = 2, embedding_size = 5, learning_rate = 0.001, epoch = 10) :
    tokenized_docs = [d.split() for d in docs]
    ohe = OneHotEncoder(tokenized_docs)
    encoded_docs = ohe.encode(tokenized_docs)
    
    onehot_size = len(encoded_docs[0][0])
    W1 = np.random.rand(onehot_size, embedding_size)
    W2 = np.random.rand(embedding_size, onehot_size)

    sliding_docs = self._slide_window(encoded_docs, win_size)
    for i in tqdm(range(epoch), desc='word embedding') :
      for d in  sliding_docs:
        for center, context in d : 
          hidden = np.dot(center, W1)
          predict = np.dot(hidden, W2)
          W1, W2 = self._backpropagation(W1, W2, hidden, predict, center, context, learning_rate)
      #self.loss += C*np.log(np.sum(np.exp(self.u))) 
    
    self.i2w = ohe.i2w
    self.wv_ = {self.i2w[i]:list(we) for i, we in enumerate(W1)}
    self.word_vectors = W1

    return self.wv_

  def most_similar(self, word, n = 3) :
    v = self.wv_[word]
    similarity = np.dot(v, self.word_vectors.T)
    return [(self.i2w[i], similarity[i], self.word_vectors[i]) for i in similarity.argsort()[::-1][:n]]

  def visualize(self, word, n = 3) :
    wvs = self.most_similar(word, n)

    # 기본 글꼴 변경
    mpl.font_manager._rebuild()
    mpl.pyplot.rc('font', family='NanumBarunGothic')

    words = [item[0] for item in wvs]
    wvs = [item[2] for item in wvs]

    tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(wvs)
    labels = words

    plt.figure(figsize=(14, 8))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')


if __name__ == "__main__" :
    docs = ["natural language processing and machine learning is fun and exciting",
            "natural language processing and machine learning"]
    
    sg = Word2Vec()
    wv = sg.embedding(docs, epoch=1000)
    print(sg.wv_['natural'])
    print(sg.most_similar('natural'))
    sg.visualize('natural', 8)