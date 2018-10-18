# data streaming and preprocessing
import numpy as np
import itertools
from sklearn import preprocessing


class dataset():

    def __init__(self, corpus, w2v=None, val_ratio=None):
        self.num_sentence = 0
        self.sentences = []
        self.w2vembedd = w2v
        self.max_length = 0
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(np.arange(6))
        '''
        with open(corpus, 'r', encoding='utf-8') as f:
            for i, l in enumerate(f):
                self.num_sentence += 1
                self.sentences.append(l)
        '''
        cut_programs = np.load('cut_Programs.npy')
        self.sentences = list(itertools.chain.from_iterable(itertools.chain.from_iterable(cut_programs)))
        self.num_sentence = len(self.sentences)

        if val_ratio is not None:
            self.split_dataset(val_ratio)
    def sentence2vec(self, sentence_idx):
        sentvec = np.empty((0, 256))
        for w in self.sentences[sentence_idx]:
            if w.isalnum():
                try:
                    sentvec = np.vstack([sentvec,self.w2vembedd.wv[w]])
                except KeyError:
                    pass    
        return sentvec
    def getbatch(self, mode='train', batch_size=64):
        if mode == 'train':
            batch_idx = self.getQApair(batch_size)
            label = np.zeros([batch_size])
            sentvec = []
            for i, QA in enumerate(batch_idx):
                label[i] = np.argwhere(QA[1:]-1 == QA[0])
                for s in QA:
                    s_vec = self.sentence2vec(s)
                sentvec.append(s_vec)
            print(s_vec[0].shape)
        return sentvec, self.lb.transform(label)
            #for i in range(batch_size):


    def split_dataset(self, val_ratio):
        # routine for splitting dataset
        val_num = int(val_ratio * self.num_sentence)
        self.tot_idx = np.arange(self.num_sentence)
        rnd_idx = np.random.randint(self.num_sentence)
        self.val_index = self.tot_idx[
            rnd_idx:np.minimum(-1, self.num_sentence - rnd_idx - val_num)]
        if self.num_sentence - rnd_idx - val_num < 0:
            self.val_index = np.append(self.val_index, self.tot_idx[
                                       :self.num_sentence - rnd_idx - val_num])
            self.train_index = self.tot_idx[
                -self.num_sentence - rnd_idx - val_num:rnd_idx]
        else:
            self.train_index = self.tot_idx[
                self.num_sentence - rnd_idx - val_num:self.num_sentence - val_num]
        return

    def getQApair(self, n=1):
        # return the indexes of 1 question and 1 answer and 5 wrong answers
        Q_idx = np.sort(np.random.choice(self.train_index[:-1], n, replace=False))
        # while the consecutive sentence appear in the idx, resampling it again
        batch_idx = np.zeros([n, 7]).astype(np.int64)
        # while np.any(np.diff(Q_idx) == 1):
            # Q_idx = np.sort(np.random.choice(self.train_index, n, replace=False))
        genAnsidx_tot = np.random.choice(self.train_index, 5 * n, replace=False)
        for i, Q in enumerate(Q_idx):
            batch_idx[i, 0] = Q
            candidate = np.array([Q+1])
            genAnsidx = genAnsidx_tot[i*5:(i+1)*5]
            while np.any(np.logical_or(genAnsidx == Q+1, genAnsidx == Q)):
                genAnsidx = np.random.choice(self.train_index, 5, replace=False)
            candidate = np.append(candidate, genAnsidx)
            batch_idx[i, 1:] = np.random.permutation(candidate)
        return batch_idx

if __name__ == '__main__':
    from gensim import models
    model = models.Word2Vec.load('word2vec.model')
    test_dataset = dataset('split_word_Programs.txt', w2v=model,val_ratio=0.2)

    for i in range(10):
        s,l = test_dataset.getbatch()
        print(s[0],l[0])
