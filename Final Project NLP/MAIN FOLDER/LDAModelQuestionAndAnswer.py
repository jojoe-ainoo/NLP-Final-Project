
# coding: utf-8

# In[93]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

# In[98]:


#restrcturing data in test file into a document-class grouping
def partitionData(trianTestFile,trianTestFileTopics):
    tr_dataset = open(trianTestFile,"r")
    tr_dataset_topics = open(trianTestFileTopics,"r")

    document_data = []
    document_class = []

    line = tr_dataset.readline()
    line2 = tr_dataset_topics.readline()

    while line:
        document_data.append(line.replace("\t","").replace("\n","").replace("\r",""))
        
        document_class.append(line2.replace("\t","").replace("\n","").replace("\r",""))
        line = tr_dataset.readline()
        line2 = tr_dataset_topics.readline()

    tr_dataset.close()
    tr_dataset_topics.close()

    return ([document_data,document_class])


# In[99]:


def vectorizer():
    
    dataframe = partitionData("../FAQs/Questions.txt","../FAQs/Answers.txt")
     
    #using rule-based to normalize the data
    #Rule 1: All words should be converted into lowercase letters
    
    #Rule 2: The encoding scheme used should be utf-8
    
    #Rule 3: The lower and upper boundary of the range of n-values 
             #for different n-grams to be extracted should be [1,1]
        
    #Rule 4: When building the vocabulary ignore terms that have a document 
             #frequency strictly higher than the given threshold which is 1.0
        
    #Rule 5: When building the vocabulary ignore terms that have a document 
             #frequency strictly lower than the given threshold which is 1
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase=True,
        encoding='utf-8',
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
    )

    data_vectorized = vectorizer.fit_transform(dataframe[0])
    features_nd = data_vectorized.toarray()


    # Convert raw frequency counts into TF-IDF values
    tfidf_transformer = TfidfTransformer()
    sent_tfidf = tfidf_transformer.fit_transform(data_vectorized).toarray()


    no_features = 1000

    # # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(dataframe[0])
    tf_feature_names = tf_vectorizer.get_feature_names()


    no_topics = 126

    # Run LDA
    lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    
    lda_Z = lda_model.fit_transform(data_vectorized)
    
    return ([dataframe[1],lda_Z,lda_model,vectorizer])


# In[100]:



#passing the test file with questions
def passTestFile(questionFile):
    
    dataframe_lda_Z = vectorizer()#calling the vectorizer to return the transformed lda_model
    lda_model = dataframe_lda_Z[2]
    vectorizerObj = dataframe_lda_Z[3]
    tr_dataset = open(questionFile,"r")

    text_questions = []
    To = open("qa_result.txt","w")
    line = tr_dataset.readline()

    while line:
        text_questions.append(line)
        line = tr_dataset.readline()

    tr_dataset.close()

    #generating and printing all appropriate topics to questions
    if len(text_questions) > 0:
        for question in text_questions:
            x = lda_model.transform(vectorizerObj.transform([question]))[0]

            def most_similar(x, Z, top_n=5):
                dists = euclidean_distances(x.reshape(1, -1), Z)
                pairs = enumerate(dists[0])
                most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
                return most_similar

            similarities = most_similar(x, dataframe_lda_Z[1])

            document_id, similarity = similarities[0]
            t = dataframe_lda_Z[0][document_id][:1000]
            To.write(t +" \n")
    To.close()







