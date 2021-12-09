import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# t = {'US/Eastern': 0, 'US/Pacific': 1, 'US/Central': 2, 'US/Mountain': 3}
s = {'Day': 0, 'Night': 1}

def one_hot(x):
    '''
    One-hot encoding method for mapping given values.
    '''
    if x >= 50:
        return 0
    return 1

dataset = pd.read_csv('update.csv')


def mapping():
    '''
    Mapping operation for too large data scale.
    Given value mapped as much smaller that matched in range or exact value.
    Sunrise_Sunset : {Day/Night} -> {0/1}
    Distance(mi) : {0-155} -> {>=50 or <50}
    '''
    # dataset['Humidity(%)'] = dataset['Humidity(%)'].map(one_hot)
    dataset['Sunrise_Sunset'] = dataset['Sunrise_Sunset'].map(s)
    dataset['Distance(mi)'] = dataset['Distance(mi)'].map(one_hot)
    # weather condition sunrise sunset timezone ===> severity?


def evaluate_metric(testY, y_pred):
    from sklearn.metrics import classification_report, accuracy_score
    report = classification_report(testY, y_pred)
    print("Classification Report:",)
    print(report)
    score = accuracy_score(testY, y_pred)
    print("Accuracy:", score)

'''
Classification Report:
              precision    recall  f1-score   support

           1       0.00      0.00      0.00      6350
           2       0.85      1.00      0.92    209266
           3       0.00      0.00      0.00     18980
           4       0.00      0.00      0.00     12621

    accuracy                           0.85    247217
   macro avg       0.21      0.25      0.23    247217
weighted avg       0.72      0.85      0.78    247217

Accuracy: 0.8464870943341275
'''
def decision_tree(trainX, testX, trainY, testY):
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(trainX, trainY)

    # print(dtree.predict([[0.5, 3, 60]]))
    y_pred = dtree.predict(testX)
    # y_score = dtree.score(testX, testY)

    evaluate_metric(testY, y_pred)

'''
Classification Report:
              precision    recall  f1-score   support

           1       0.00      0.00      0.00      6622
           2       0.84      1.00      0.92    208869
           3       0.00      0.00      0.00     19044
           4       0.00      0.00      0.00     12682

    accuracy                           0.84    247217
   macro avg       0.21      0.25      0.23    247217
weighted avg       0.71      0.84      0.77    247217

Accuracy: 0.8448812177156102
'''
def random_forest(trainX, testX, trainY, testY):
    classifier = RandomForestClassifier(n_estimators = 50)
    classifier.fit(trainX, trainY)

    y_pred = classifier.predict(testX)
    
    evaluate_metric(testY, y_pred)

'''
Classification Report:
              precision    recall  f1-score   support

           1       0.00      0.00      0.00      6488
           2       0.85      1.00      0.92    209174
           3       0.00      0.00      0.00     19127
           4       0.00      0.00      0.00     12428

    accuracy                           0.85    247217
   macro avg       0.21      0.25      0.23    247217
weighted avg       0.72      0.85      0.78    247217

Accuracy: 0.8461149516416752
'''
def logistic_regression(trainX, testX, trainY, testY):
    digreg = linear_model.LogisticRegression()
    digreg.fit(trainX, trainY)

    y_pred = digreg.predict(testX)

    evaluate_metric(testY, y_pred)


def classification(method):
    features = ['Distance(mi)', 'Sunrise_Sunset']
    mapping()

    X = dataset[features]
    Y = dataset['Severity']

    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25)

    if method == 'decision':
        decision_tree(trainX, testX, trainY, testY)
    elif method == 'random':
        random_forest(trainX, testX, trainY, testY)
    else:
        logistic_regression(trainX, testX, trainY, testY)


if __name__ == '__main__':
    classification('regression')