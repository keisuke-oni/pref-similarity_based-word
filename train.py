from gensim.models import word2vec
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence(sys.argv[1])
model = word2vec.Word2Vec(sentences,
                          sg=1,
                          size=200,
                          min_count=3,
                          window=8,
                          hs=1)
model.save(sys.argv[2])
