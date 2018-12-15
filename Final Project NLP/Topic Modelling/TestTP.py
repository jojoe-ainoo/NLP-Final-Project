#  Code to Test


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

# use title array and implement this
def WriteResults(results,Model):
    file = open("results-"+Model+".txt", "w")
    for result in results:
        file.write(result)
        file.write("\n")
