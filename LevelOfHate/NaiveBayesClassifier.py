from collections import defaultdict
import numpy as np

class NaiveBayesClassifier(object):
        prior = defaultdict(int)
        logPrior = {}
        bigDoc = defaultdict(list)
        logLikelihoods = defaultdict(defaultdict)
        V = []
        n = 1
        word_count = defaultdict(int)

        def _init_(self, n_gram=1):
            self.prior = defaultdict(int)
            self.logPrior = {}
            self.bigDoc = defaultdict(list)
            self.logLikelihoods = defaultdict(defaultdict)
            self.V = []
            self.n = n_gram


        def compute_vocabulary(self, documents):
            vocabulary = set()

            for doc in documents:
                for word in doc.split(" "):
                    vocabulary.add(word)

            return vocabulary

        def count_word_in_classes(self):
            counts = {}
            for c in list(self.bigDoc.keys()):
                docs = self.bigDoc[c]
                counts[c] = defaultdict(int)
                for doc in docs:
                    words = doc.split(" ")
                    for word in words:
                        counts[c][word] += 1
            return counts

        def compute_prior_and_bigdoc(self, training_set, training_labels):
            grams = 1
            for x, y in zip(training_set, training_labels):
                all_words = x.split(" ")
                if self.n == 1:
                    grams = all_words
                # else
                #     grams=self.words_to_grams(all_words)

                self.prior[y] += len(grams)
                self.bigDoc[y].append(x)

        def train(self, training_set, training_labels, alpha=1):
            # Get number of documents
            N_doc = len(training_set)

            # Get vocabulary used in training set
            self.V = self.compute_vocabulary(training_set)

            # Create bigdoc
            for x, y in zip(training_set, training_labels):
                self.bigDoc[y].append(x)

            # Get set of all classes
            all_classes = set(training_labels)

            # Compute a dictionary with all word counts for each class
            self.word_count = self.count_word_in_classes()

            # For each class
            for c in all_classes:
                # Get number of documents for that class
                N_c = float(sum(training_labels == c))

                # Compute logprior for class
                self.logPrior[c] = np.log(N_c / N_doc)

                # Calculate the sum of counts of words in current class
                total_count = 0
                for word in self.V:
                    total_count += self.word_count[c][word]

                # For every word, get the count and compute the log-likelihood for this class
                for word in self.V:
                    count = self.word_count[c][word]
                    self.logLikelihoods[c][word] = np.log((count + alpha) / (total_count + alpha * len(self.V)))

        def getHateLevel(self, test_doc):
            sums = {
                0: 0,   # Neg
                1: 0,   # Pos
            }
            # print (self.bigDoc.keys())
            for c in self.bigDoc.keys():
                sums[c] = self.logPrior[c]
                words = test_doc.split(" ")
                for word in words:
                    if word in self.V:
                        sums[c] += self.logLikelihoods[c][word]
            # print("Positive: "+ str(sums[1]))
            # print("Negative: "+ str(sums[0]))
            sum = sums[0]+sums[1]
            hate_level = float(sums[0]/sum)
            return hate_level