import nltk, re
from nltk.corpus import stopwords
from gensim.models import LdaModel
from collections import defaultdict
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

# Here set the number of topics(to be changed if necessary)
nb=10
  
id2word = Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]

lda = LdaModel(
  corpus, 
  num_topics=nb, 
  id2word=id2word, 
  passes=1000,
  alpha='auto', 
  eta='auto',
  decay=0.5, 
  offset=1.0
)

# Print topic descrition
for i in range(0, nb):
  value=lda.get_topic_terms(i)
  print("Topic ", i+1)
  for j in value:
    word = id2word[j[0]]
    print(f"P({word}) = {j[1]}")
  print()

# Compute Perplexity, a measure of how good the model is (lower the better).
perplexity_lda=lda.log_perplexity(corpus)  
print(f'Perplexity = {perplexity_lda}')

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence = {coherence_lda}\n')

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
    print(f"Most typical document for topic {topic_id+1} is document {doc_index+1} with proportion {proportion}")
