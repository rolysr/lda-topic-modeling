import nltk
import matplotlib.pyplot as pl

#Given an object named 'texts' (a list of tokenized texts, ie a list of lists of tokens)
#from gensim.test.utils import common_texts as texts

# Create a corpus from a list of lists of tokens
from gensim.corpora.dictionary import Dictionary

from nltk.corpus import stopwords
nltk.download('stopwords')

texts=[]
file = open("TokenVieuxN.txt", "r")
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

# Set variables for plotting coherence against number of topic
X_nb = []
Y_coherence = []

#Here set the number of topics(to be changed if necessary)
for nb in range(5, 31):
  X_nb.append(nb) # Add nb value to X axis

  id2word = Dictionary(texts)
  corpus = [id2word.doc2bow(text) for text in texts]
  #print(corpus)

  # Print dictionnary
  # for i in id2word:
  #   print(i, id2word[i])

  # Train the lda model on the corpus.
  from gensim.models import LdaModel
  lda = LdaModel(corpus, num_topics=nb)

  # Print topic descrition
  # for i in range(0, nb-1):
  #   value=lda.get_topic_terms(i)
  # #  print(value)
  #   print("Topic ", i)
  #   for j in value:
  #     print(id2word[j[0]], " - P=", j[1])
  #   print()

  # Compute Perplexity
  perplexity_lda=lda.log_perplexity(corpus)  # a measure of how good the model is (lower the better).
  # print('Perplexity= ', perplexity_lda)

  # Compute Coherence Score
  from gensim.models import CoherenceModel
  coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  # print('Coherence= ', coherence_lda)

  Y_coherence.append(coherence_lda) # Add coherence to Y axis

# Plot coherence against nb
print(X_nb)
print(Y_coherence)
pl.title("Number of Topics - Coherence LDA")
pl.plot(X_nb, Y_coherence)
pl.xlabel('Number of Topics')
pl.ylabel('Coherence LDA')
pl.show()