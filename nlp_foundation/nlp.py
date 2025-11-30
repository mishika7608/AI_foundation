 # 1. LOGISTIC REGRESSION
#shuffle data (sample func index again)-> vectorization(text to nos) -> bag of words approach -> transform sentence to number -> pandas dataframe ->split data into taining and testing data (features(ip=x)(bag of word sentences),labels(op=y)(sentiment) random()-ensures same split everytime) -> ->train model (fit-search for patterns) ->test-preict()
#precision- of all predicted(+/-) which portion was correct
#recall- of all truly (+/-) what was correctly found
#fl-score- 1 number that combines precision and recall (low-poor balance bw precision and recall)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

data = pd.DataFrame([
    ("I really enjoyed this movie it was fantastic", "positive"),
    ("The food tasted wonderful", "positive"),
    ("This product exceeded my expectations", "positive"),
    ("I feel great about this decision", "positive"),
    ("The book was incredibly inspiring", "positive"),

    ("I hated this movie it was terrible", "negative"),
    ("The service at the restaurant was poor", "negative"),
    ("I am disappointed with the results", "negative"),
    ("This app is confusing and hard to use", "negative"),
    ("The food tasted unpleasant", "negative"),
    ],
    columns=['text','sentiment'])
data = data.sample(frac=1).reset_index(drop=True)
X= data['text']
Y= data['sentiment']
countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(X)
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns= countvec.get_feature_names_out())
print(bag_of_words)
X_train, X_test, Y_train, Y_test = train_test_split(bag_of_words, Y, test_size=0.3,random_state=7)
lr = LogisticRegression(random_state=1).fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)
accuracy_score(y_pred_lr, Y_test)
print(classification_report(Y_test, y_pred_lr,zero_division=0))

#Naive-Bayes = simple classifier works on probab. checks how often word appears in a group. combine mathematically. it assumes each word is independent of other
nb = MultinomialNB().fit(X_train, Y_train)
y_pred_nb =nb.predict(X_test)
print(accuracy_score(y_pred_nb, Y_test))

# 3. Linear Support Vector Machine - finds fastest possible boundary that seperates the class (2 vals- straight line)- suppport vectors(closest to line) Multiple fearture-hyperplane
svm = SGDClassifier().fit(X_train, Y_train)
y_pred_svm = svm.predict(X_test)
print(accuracy_score(y_pred_svm, Y_test)) #still needs cleaning to inc accuracy
