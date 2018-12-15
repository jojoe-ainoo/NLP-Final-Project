# Emmanuel Jojoe Ainoo (MY NAIVE BAYES CLASSIFIER and LOGISTICS REGRESSION CLASSFIER)

import sys
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Initializing and Managing Datasets
print("---> Initializing and Managing Datasets")
print(" ")
# # This Function Separates Sentences from Classes
# def ManageTrainData(train_files):
#     data , data_labels = [] , [] # Preprocessing Data(Separate Sentences from Classes)
#     for file in train_files: # For each training file
#         with open(file) as f: # Loop through each the file
#             for i in f:
#                 data.append(i[:-1]) # Put each sentence in data array
#                 if i[-2] == "1": # Identify all positive classes
#                     data_labels.append('1')
#                 else:
#                     data_labels.append('0') # Identify negative classes
#     return[data,data_labels]

def ManageTrainData(trainDoc):
    data = []
    with open(trainDoc) as f:
        for doc in f:
            data.append(doc)
    return data

def ManageTrainClass(trainClass):
    data_labels = []
    with open(trainClass) as file:
        for c in file:
            data_labels.append(c)
    return data_labels



# Vectorizing Data for Normalized
print("---> Transforming/Vectorizing Data(Normalized Version) ")
print(" ")
#Function to Transform data into into counts and normalize it
def VectorizeNorm(data):
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = True,
    )
    features = vectorizer.fit_transform(data)
    features_nd = features.toarray()

    # Convert raw frequency counts into TF-IDF values
    tfidf_transformer = TfidfTransformer()
    sent_tfidf = tfidf_transformer.fit_transform(features).toarray()
    return[features,features_nd,tfidf_transformer,sent_tfidf,vectorizer]
#
# Spliting Data for Training and Testing
print("---> Spliting Data for Training and Testing ")
print(" ")

def Split(features_nd,data_labels):
    X_train, X_test, y_train, y_test  = train_test_split(
            features_nd,
            data_labels,
            train_size=0.80,
            random_state=1234)
    return[X_train, X_test, y_train, y_test]

# Train a Logistics Regression classifier
print("---> Training Logistics Regression Classifier ")
print(" ")

def LogisticsRegressionTrainer(X_train, y_train):
    log_model = LogisticRegression()
    log_model = log_model.fit(X=X_train, y=y_train)# Call Train Data on Naive Bayes
    return log_model

# Predicting the Test set results, find accuracy
print(" ")
print("---> Predicting Test set Results ")
def PredictResults(model,X_test,y_test):
    y_pred = model.predict(X_test)
    sklearn.metrics.accuracy_score(y_test, y_pred)
    evaluate = sklearn.metrics.accuracy_score(y_test, y_pred)
    return [y_pred,evaluate]

def TestClassifer(testfile,vectorizer,tfidf_transformer,model):
    reviews_new = []
    with open(testfile) as f:
        for i in f:
            reviews_new.append(i[:-1])

    reviews_new_counts = vectorizer.transform(reviews_new)
    reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)

    # Have classifier make a prediction
    print("   ")
    print("---> Making Prediction")
    pred = model.predict(reviews_new_tfidf)
    return pred
#
# # def WriteToFile():
# # use title array and implement this
# def WriteResults(results,classType,version):
#     file = open("results-"+classType+version+".txt", "w")
#     for result in results:
#         file.write(result)
#         file.write("\n")
#
# #Function to evaluate the Models
def Evaluate(evaluate):
    return evaluate
def main():
    print("Starting")
    traindoc = "../FAQs/Questions.txt"
    trainClass = "../FAQs/Topics.txt"

    data = ManageTrainData(traindoc)
    classes = ManageTrainClass(trainClass)
    vector = VectorizeNorm(data)

    split = Split(vector[1],classes)
    trainLRModel = LogisticsRegressionTrainer(split[0], split[2])
    predLR = PredictResults(trainLRModel,split[1],split[3])
    logEv =  Evaluate(predLR[1])
    print(logEv)




main()

#
# if __name__ == "__main__":
#     # Loop through folder and pick all files for training data
#     TrainFiles = ["sentiment labelled sentences/amazon_cells_labelled.txt","sentiment labelled sentences/yelp_labelled.txt"]
#     data = ManageTrainData(TrainFiles)
#     file = sys.argv[3]
#     #
#     # if(sys.argv[1] == "nb"):
#     #     print(" ")
#     #     print("------Running Naive Bayes Classifier------ ")
#     #     headings = [] # Get Headings to Results File
#     #     # headings.append("------Running Naive Bayes Classifier------ ")
#     #     print(" ")
#     #
#     #     split = 0 #To store values of splitted data into Test and Training
#     #     vector = 0 #Store Transformed counts
#     #     versionVal = "" #To store version of Model (nb or lr)
#     #
#     #
#     #     if(sys.argv[2] == "u"):
#     #         print("------Running UnNormalized Version ------ ")
#     #         print(" ")
#     #         # headings.append("------Running UnNormalized Version ------ ")
#     #         vector = VectorizeUnNorm(data) #Call Unnormalized Version
#     #         split = Split(vector[1],data[1])
#     #         versionVal = "u"
#     #
#     #     elif(sys.argv[2] == "n"):
#     #         print("------Running Normalized Version ------ ")
#     #         # headings.append("------Running Normalized Version ------ ")
#     #         print(" ")
#     #         vector = VectorizeNorm(data) #Call Normalized Version
#     #         split = Split(vector[1],data[1])
#     #         versionVal = "n"
#     #
#     #     trainNBModel = NaiveBayesTrainer(split[0], split[2])
#     #     predNB = PredictResults(trainNBModel,split[1],split[3])
#     #     cmNB = ConfusionMatrix(trainNBModel,split[1],split[3],predNB[0])
#     #     # file = "imdb_labelled.txt"
#     #
#     #     naiveTest = TestClassifer(file,vector[4],vector[2],trainNBModel)
#     #     print(naiveTest)
#     #     WriteResults(naiveTest,"nb-",versionVal)
#     #     naiveEv = Evaluate(predNB[1])
#     #     print("Score Naive: ",naiveEv)
#     #
#     #
#     #
#     # elif(sys.argv[1] == "lr"):
#     #     print(" ")
#     #     print("------Running Logisitics Regression Classifier------ ")
#     #     headings = []
#     #     # headings.append("------Running Logisitics Regression Classifier------ ")
#     #     print(" ")
#     #
#     #     split = 0
#     #     vector = 0
#     #     versionVal = ""
#     #
#     #     if(sys.argv[2] == "u"):
#     #         print("------Running UnNormalized Version ------ ")
#     #         # headings.append("------Running UnNormalized Version ------ ")
#     #         print(" ")
#     #         vector = VectorizeUnNorm(data)
#     #         split = Split(vector[1],data[1])
#     #         versionVal = "u"
#     #
#     #     elif(sys.argv[2] == "n"):
#     #         print("------Running Normalized Version ------ ")
#     #         # headings.append("------Running Normalized Version ------ ")
#     #         print(" ")
#     #         vector = VectorizeNorm(data)
#     #         split = Split(vector[1],data[1])
#     #         versionVal = "n"
#     #
#     #
#     #     trainLRModel = LogisticsRegressionTrainer(split[0], split[2])
#     #     predLR = PredictResults(trainLRModel,split[1],split[3])
#     #     cmLR = ConfusionMatrix(trainLRModel,split[1],split[3],predLR[0])
#     #
#     #     # file = "imdb_labelled.txt"
#     #     logTest = TestClassifer(file,vector[4],vector[2],trainLRModel)
#     #     print(logTest)
#     #     WriteResults(logTest,"lr-",versionVal)
#     #     logEv =  Evaluate(predLR[1])
#     #     print("Score Logistics: ",logEv)
