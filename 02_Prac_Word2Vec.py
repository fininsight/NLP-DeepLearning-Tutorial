from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import numpy as np

class Word2Vec() :
    def __init__(self,
                docs,                  # 학습시킬 문장
                embedding_size,             # 임베딩된 단어 벡터의 차원 크기
                learning_rate = 0.1,        # 학습률(Learning rate)
                min_count=0,                # 2번 미만 등장한 단어는 제외
                batch_size = 100,           #batc
                window = 1,                 # 문맥의 크기 (window_size)
                sg = 0,                     # 0: CBOW, 1: Skip-gram
                epoch = 10,                     
                negative_sampling = 0,       # negative sample 갯수. 0은 적용 않함
                ) :

        self.tokenized_docs = [d.split() for d in docs]
        self.embedding_size = embedding_size
        self.learning_rate =learning_rate
        self.min_count = min_count
        self.window = window
        self.sg = sg
        self.epoch = epoch
        self.negative_sampling = negative_sampling

        self.w2i = defaultdict(lambda:len(self.w2i))
        [self.w2i[w] for d in self.tokenized_docs for w in d]
        self.i2w = {i:w for w,i in self.w2i.items()}

        X, Y = self._gen_window_data()
        self._optimize(X, Y)

    def wv(self, word) :
        return self.WV[self.w2i[word]]

    def infernce(self, w1, w2, w3, n=3):
        w1v = self.wv(w1)
        w2v = self.wv(w2)
        w3v = self.wv(w3)

        inference_v = w1v - w2v + w3v
        return self._most_similar(inference_v, n)

    def _most_similar(self, v, n = 3):
        return np.dot(v, self.WV)

    def most_similar(self, word, n = 3):
        return self._most_similar(self.wv(word))

    def _gen_window_data(self):
        context, center = [], []
        context_onehot, center_onehot = [], []
        for d in self.tokenized_docs : 
            for i, w in enumerate(d) :
                s, e = max(0, i - self.window), min(len(d), i + self.window)
                tmp = [self.w2i[w] for w in d[s:e+1]]
                c = tmp.pop(i-s)
                co = [0] * len(self.w2i)
                center.append(c)
                co[c] = 1
                center_onehot.append(co)
                context.append(tmp)

        if self.sg == 0 :
            return context, np.array(center_onehot)
        else :
            return center, context

    def _softmax(self, output) :
        return np.exp(output) / np.sum(np.exp(output), keepdims=True, axis=0)

    def _sigmoid (self, o):
        return 1./(1+np.exp(-1*o))

    def _init_weights(self, n_words):
        W1 = np.random.rand(n_words, self.embedding_size)
        W2 = np.random.rand(self.embedding_size, n_words)

        return W1, W2

    def _input_to_hidden(self, X, W) :
        H = np.vstack([np.mean(W[x], keepdims=True, axis=0) for x in X]).T
        return H

    def _hidden_to_output(self, H, W) :
        if self.negative_sampling == 0 :
            Y_hat = self._softmax(np.dot(W.T, H))

        return Y_hat.T

    def _cal_gradients(self, X, Y, Y_hat, H, W2) :
        err = Y_hat-Y
        
        dW2 = np.dot(H, err)
        dW1 = np.outer(X, np.dot(W2, err.T))
        
        return dW1, dW2

    def _eval_loss (self, Y, Y_hat):
        return -1/len(Y)*np.sum(Y*np.log(Y_hat))

    def _optimize(self, X, Y):
        W = self._init_weights(len(self.w2i))
        loss_trace = []

        for e in range(self.epoch) :
            H = self._input_to_hidden(X, W[0])
            Y_hat = self._hidden_to_output(H, W[1])
            loss = self._eval_loss(Y, Y_hat)
            gradients = self._cal_gradients(X, Y, Y_hat, H, W[1])

            for w, gradient in zip(W, gradients):
                w += - self.learning_rate * gradient

            loss_trace.append(loss)
        
        self.WV = W[0]



if __name__ == "__main__" :
    docs = ["natural language processing and machine learning is fun and exciting",
            "natural language processing and machine learning"]

    wv = Word2Vec(docs, embedding_size=2)
    