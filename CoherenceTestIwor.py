import nltk

#Given an object named 'texts' (a list of tokenized texts, ie a list of lists of tokens)
#from gensim.test.utils import common_texts as texts

# Create a corpus from a list of lists of tokens
from gensim.corpora.dictionary import Dictionary

from nltk.corpus import stopwords
from gensim.matutils import cossim, sparse2full
import numpy as np
nltk.download('stopwords')

texts=[]
file = open("TokenVieuxM.txt", "r")
lines = file.readlines()
file.close()

#print(stopwords.words())

for line in lines:
  line=line.strip()
  lt=line.split(",")
#Potential ill-character cleaning
  for i in range(len(lt)):
    lt[i]=lt[i].replace('[','')
    lt[i]=lt[i].replace(']','')
    lt[i]=lt[i].replace('"','')
    lt[i]=lt[i].replace('\n','')
    lt[i]=lt[i].replace(' ', '')
#End : Potential ill-characters cleaning
# print(lt)
  ltc=[word for word in lt if not word in stopwords.words()]
#  print("C", ltc)
  texts.append(ltc)

#for text in texts:
  #print(text)

#Here set the number of topics(to be changed if necessary)
nb=10
  
id2word = Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]
#print(corpus)

# Print dictionnary
for i in id2word:
  print(i, id2word[i])

# Train the lda model on the corpus.
from gensim.models import LdaModel
lda = LdaModel(corpus, num_topics=nb)

# Print topic descrition
for i in range(0, nb-1):
  value=lda.get_topic_terms(i)
#  print(value)
  print("Topic ", i)
  for j in value:
    print(id2word[j[0]], " - P=", j[1])
  print()

# Compute Perplexity
perplexity_lda=lda.log_perplexity(corpus)  # a measure of how good the model is (lower the better).
print('Perplexity= ', perplexity_lda)

# Compute Coherence Score
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence= ', coherence_lda)

# Find the most typical document for each topic
# additional import
from collections import defaultdict

# ... your existing code ...

# Compute topic proportions for each document
doc_topics = lda.get_document_topics(corpus, minimum_probability=0)

# Initialize a dictionary to store the most typical document for each topic
most_typical_docs = defaultdict(lambda: (-1, -1)) # (doc_index, probability)

# Iterate over documents
for doc_index, doc_topic_proportions in enumerate(doc_topics):
    # Iterate over topic proportions
    for topic_id, proportion in doc_topic_proportions:
        # If this document has a higher proportion for this topic than the current most typical document, update it
        if proportion > most_typical_docs[topic_id][1]:
            most_typical_docs[topic_id] = (doc_index, proportion)

# Print most typical documents for each topic
for topic_id, (doc_index, proportion) in most_typical_docs.items():
    print(f"Most typical document for topic {topic_id} is document {doc_index} with proportion {proportion}")
