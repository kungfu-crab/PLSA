import numpy as np
import pandas as pd
from collections import Counter
import json
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# segmentation, stopwords filtering and document-word matrix generating
def preprocessing(datasetFilePath, stopwordsFilePath):
    # read the stopwords file
    with open(stopwordsFilePath, 'r') as file:
        stopwords = set([line.strip() for line in file])

    # read the documents
    documents = list(pd.read_csv(datasetFilePath)['ABSTRACT'])
    documents = random.sample(documents, 1000)

    words = set()
    word_counts = []
    for document in documents:
        seglist = word_tokenize(document)
        wordlist = []
        for word in seglist:
            synsets = wordnet.synsets(word)
            if synsets:
                syn_word = synsets[0].lemmas()[0].name()
                if syn_word not in stopwords:
                    wordlist.append(syn_word)
            else:
                if word not in stopwords:
                    wordlist.append(word)
        words = words.union(wordlist)
        word_counts.append(Counter(wordlist))
    word2id = {words:id for id, words in enumerate(words)}
    id2word = dict(enumerate(words))

    N = len(documents) # number of documents
    M = len(words) # number of words
    X = np.zeros((N, M))
    for i in range(N):
        for keys in word_counts[i]:
            X[i, word2id[keys]] = word_counts[i][keys]
    print(f"documents count: {N}")
    print(f"words count: {M}")

    return N, M, word2id, id2word, X

def E_step(lam, theta):
    # lam: N * K, theta: K * M, p = K * N * M
    N = lam.shape[0]
    M = theta.shape[1]
    lam_reshaped = np.tile(lam, (M, 1, 1)).transpose((2,1,0)) # K * N * M
    theta_reshaped = np.tile(theta, (N, 1, 1)).transpose((1,0,2)) # K * N * M
    temp = lam @ theta
    p = lam_reshaped * theta_reshaped / temp
    return p

def Initialize(lam, theta):
    # lam: N * K, theta: K * M
    lam = lam / np.sum(lam, axis=1)[:, np.newaxis]
    theta = theta / np.sum(theta, axis=1)[:, np.newaxis]
    return lam, theta

def M_step(p, X):
    # p: K * N * M, X: N * M, lam: N * K, theta: K * M
    # update lam
    lam = np.sum(p * X, axis=2) # K * N
    lam = lam / np.sum(lam, axis=0) # normalization for each column
    lam = lam.transpose((1,0)) # N * K

    # update theta
    theta = np.sum(p * X, axis=1) # K * M
    theta = theta / np.sum(theta, axis=1)[:, np.newaxis] # normalization for each row
    
    return lam, theta

def LogLikelihood(p, X, lam, theta):
    # p: K * N * M, X: N * M, lam: N * K, theta: K * M
    res = np.sum(X * np.log(lam @ theta)) # N * M
    return res

def main():
    datasetFilePath = 'Task-Corpus.csv'
    stopwordsFilePath = 'stopwords.dic'
    K = 4    # number of topic
    maxIteration = 200
    threshold = 10.0
    topicWordsNum = 10
    
    N, M, word2id, id2word, X = preprocessing(datasetFilePath, stopwordsFilePath)

    # lam[j, k] = p(z_k | d_j), lam: N * K
    lam = np.random.rand(N, K)

    # theta[k, i] = p(w_i | z_k), theta: K * M
    theta = np.random.rand(K, M)
    
    # p[k, i, j] = p(z_k | w_i, d_j), p: K * N * M
    p = np.zeros((K, N, M))
    
    lam, theta = Initialize(lam, theta)
    loglikelihood = -np.inf
    for iter in range(maxIteration):
        old_loglikelihood = loglikelihood
        p = E_step(lam, theta)
        lam, theta = M_step(p, X)
        loglikelihood = LogLikelihood(p, X, lam, theta)
        print(f"iter {iter + 1}, loglikelihood={loglikelihood}")
        if np.isnan(loglikelihood) or loglikelihood - old_loglikelihood < threshold:
            break
    
    # output
    with open('topics.txt', 'w') as f:
        index = np.argsort(theta, axis=1)[:, :-topicWordsNum:-1]
        for i in index:
            f.write(str([id2word[j] for j in i]) + '\n')

    with open('dictionary.json', 'w') as f:
        json.dump(word2id, f)

    np.savetxt('DocTopicDistribution.csv', lam, delimiter=',')
    np.savetxt('TopicWordDistribution.csv', theta, delimiter=',')

if __name__ == "__main__":
    main()