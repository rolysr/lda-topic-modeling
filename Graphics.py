import nltk, re
import matplotlib.pyplot as pl
from nltk.corpus import stopwords
from gensim.models import LdaModel
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# load corpus
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
print()

texts=[]

# Load document
file = open("datasets/TokenVieuxM.txt", "r")
lines = file.readlines()
file.close()

# Initialize lemmatizer
lemmer = WordNetLemmatizer()

for line in lines:
  # Parsing text and removing unwanted characters
  line = re.findall(r"[\w]+", line)

  # Remove stopwords  
  words = [word for word in line if not word in stopwords.words()]

  # Function to test if something is a noun
  is_noun = lambda pos: pos[:2] == 'NN'

  # Tag the tokens with POS tags
  nouns = [ lemmer.lemmatize(word)
    for (word, pos) in nltk.pos_tag(words) if is_noun(pos)] 

  texts.append(nouns)

# Set variables for plotting coherence against number of topic
X_nb = []
Y_coherence = []

# Here set the number of topics(to be changed if necessary)
for nb in range(5, 31):
  X_nb.append(nb) # Add nb value to X axis

  id2word = Dictionary(texts)
  corpus = [id2word.doc2bow(text) for text in texts]

  lda = LdaModel(
    corpus, 
    num_topics=nb, 
    id2word=id2word, 
    passes=1000,
    alpha='auto', 
    eta='auto',
    decay=0.25, 
    offset=1.0
  )

  # Compute Perplexity, a measure of how good the model is (lower the better).
  perplexity_lda=lda.log_perplexity(corpus)  

  # Compute Coherence Score
  coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()

  # Add coherence to Y axis
  Y_coherence.append(coherence_lda) 

# Plot coherence against nb
print(X_nb)
print(Y_coherence)
pl.title("Number of Topics - Coherence LDA")
pl.plot(X_nb, Y_coherence)
pl.xlabel('Number of Topics')
pl.ylabel('Coherence LDA')
pl.show()