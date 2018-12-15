
# coding: utf-8

# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances


# In[3]:


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])


# In[4]:


#restrcturing data in test file into a document-class grouping
def partitionData(trianTestFile,trianTestFileTopics):
    tr_dataset = open(trianTestFile,"r")
    tr_dataset_topics = open(trianTestFileTopics,"r")

    document_data = []
    document_class = []

    line = tr_dataset.readline()
    line2 = tr_dataset_topics.readline()

    while line:
        document_data.append(line)
        document_class.append(line2)
        line = tr_dataset.readline()
        line2 = tr_dataset_topics.readline()

    tr_dataset.close()
    tr_dataset_topics.close()

    return ([document_data,document_class])


# In[5]:
dataframe = partitionData("../FAQs/Questions.txt","../FAQs/Topics.txt")


vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)

data_vectorized = vectorizer.fit_transform(dataframe[0])
features_nd = data_vectorized.toarray()


# Convert raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
sent_tfidf = tfidf_transformer.fit_transform(data_vectorized).toarray()

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd,
        dataframe[1],
        train_size=0.80,
        random_state=1234)


# In[15]:


no_features = 1000

# # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(dataframe[0])
tf_feature_names = tf_vectorizer.get_feature_names()


no_topics = 126

# Run LDA
lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_Z = lda_model.fit_transform(data_vectorized)

# Build a Non-Negative Matrix Factorization Model
nmf_model = NMF(n_components=no_topics)
nmf_Z = nmf_model.fit_transform(data_vectorized)


# In[9]:


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])


# In[16]:

#passing the test file with questions
def passTestFile(questionFile):
    tr_dataset = open(questionFile,"r")

    document_data = []

    line = tr_dataset.readline()

    while line:
        document_data.append(line)
        line = tr_dataset.readline()

    tr_dataset.close()
    tr_dataset_topics.close()

    return (document_data)

text_questions = passTestFile("")

#generating and printing all appropriate topics to questions
for in text_questions:
    x = lda_model.transform(vectorizer.transform([text_questions]))[0]

    def most_similar(x, Z, top_n=5):
        dists = euclidean_distances(x.reshape(1, -1), Z)
        pairs = enumerate(dists[0])
        most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
        return most_similar

    similarities = most_similar(x, lda_Z)

    document_id, similarity = similarities[0]
    print(dataframe[1][document_id][:1000])
