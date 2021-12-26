import pandas as pd
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

from utils import *

input_file_1 = 'data/numerical_train.csv'
input_file_2 = 'data/categorical_train.csv'
predictionClass = 'Severity'
feature_classes_1 = ['Temperature(F)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
feature_classes_2 = ['Junction', 'Traffic_Signal', 'Crossing', 'Weather_Condition', 'Sunrise_Sunset']


def drop_outlier_v2(data):
    """
    removes outlier from data
    :param data: pandas dataframe
    :return: pandas dataframe with removed outliers
    """
    for i in range(len(data.columns)):
        if type(data.iloc[0, i]) in [np.int64, np.int, np.int32, np.float64, np.float32, float, int]:
            z_score = stats.zscore(data.iloc[:, i])
            for x in range(len(data.iloc[:, i])):
                if np.abs(z_score[x]) > 3:
                    data.iloc[x, i] = np.nan

    return data.dropna().reset_index(drop=True)


def get_data(input_file, classes, prediction_class='Severity', is_normalize=False, remove_outlier=False):
    '''
    loads data from file system and prepares for training
    :param remove_outlier: if true drops outlier values, if feature classes contains NON NUMERICAL values,
     change this with FALSE
    :param input_file: input csv file
    :param classes: feature classes
    :param prediction_class: prediction class
    :param is_normalize: if True normalizes numerical values, if feature classes contains NON NUMERICAL values,
     change this with FALSE
    :return: returns train and test values
    '''
    df = pd.read_csv(input_file)
    if remove_outlier:
        df = drop_outlier(df)
    # df = df[df.Severity != 4]
    df.dropna()

    X = df[classes]
    y = df[prediction_class]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True)

    if is_normalize:
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)

    return X_train, X_test, y_train, y_test


def get_statistics(real_data, predicted_data):
    """
    returns statistics of model
    :param real_data: real prediction class data exp: Y_test
    :param predicted_data: predicted  prediction class data
    :return:
    """
    report = classification_report(real_data, predicted_data)
    print("Classification Report:", )
    print(report)

    score = accuracy_score(real_data, predicted_data)
    print("Accuracy:", score)

    cm = confusion_matrix(real_data, predicted_data, normalize='true')
    print(cm)
    cm_df = pd.DataFrame(cm, columns=[1, 2, 3, 4], index=[1, 2, 3, 4])
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    heatmap(cm_df, cmap='Blues', annot=True)
    plt.show()


def train_classifier(data_list, prediction_method='logistic_regression',
                     saved_model_name=None, get_stats=True):
    """
    trains selected classifier
    :param data_list: data_list[0] = X_train, data_list[1] = X_test, data_list[2] = Y_train, data_list[3] = Y_test
    :param prediction_method: classifier method exp: KNN
    :param saved_model_name: save model if is not NONE with name
    :param get_stats: gets statistics of fitted model
    :return:
    """
    if prediction_method == 'logistic_regression':
        model = LogisticRegression(solver='lbfgs', max_iter=100)

    elif prediction_method == 'decision_tree':
        model = DecisionTreeClassifier()

    elif prediction_method == 'knn':
        model = KNeighborsClassifier(n_neighbors=1)

    else:
        print('Please give correct classifier name !')
        return

    model.fit(data_list[0], data_list[2])

    print(f'Training of {prediction_method} classifier completed !')
    if get_stats:
        print(f'Statistics of {prediction_method} classifier:')
        get_statistics(real_data=data_list[3], predicted_data=model.predict(data_list[1]))

    # print('Accuracy of {model_name} classifier on training set: {:.2f}'
    #       .format(model.score(data_list[0], data_list[2]), model_name=prediction_method))
    #
    # print('Accuracy of {model_name} classifier on test set: {:.2f}'
    #       .format(model.score(data_list[1], data_list[3]), model_name=prediction_method))

    if saved_model_name is not None:
        save_model(model=model, model_name=saved_model_name)
        print(f'Model saved successfully with name {saved_model_name}')

    return


def prediction(model, x_test, y_test):
    """
    make prediction
    :param model: classifier (prediction) model
    :param x_test: values
    :param y_test: classes that belongs to values
    :return:
    """
    print(f'Predicted Values of Test Set = {model.predict(x_test).tolist()}')
    print(f'Actual Values of Test Set    = {y_test.tolist()}')
    pass


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data(input_file=input_file_1, classes=feature_classes_1,
                                                prediction_class=predictionClass, is_normalize=True,
                                                remove_outlier=True)
    dataList = [X_train, X_test, y_train, y_test]

    train_classifier(data_list=dataList, prediction_method='decision_tree', saved_model_name='cat_decision')
    train_classifier(data_list=dataList, prediction_method='knn', saved_model_name='cat_knn')
    train_classifier(data_list=dataList, prediction_method='logistic_regression', saved_model_name='cat_log_reg')

    X_train_2, X_test_2, y_train_2, y_test_2 = get_data(input_file=input_file_2, classes=feature_classes_2,
                                                        prediction_class=predictionClass, is_normalize=False,
                                                        remove_outlier=False)
    dataList_2 = [X_train_2, X_test_2, y_train_2, y_test_2]

    train_classifier(data_list=dataList, prediction_method='logistic_regression', saved_model_name='num_logreg')
    train_classifier(data_list=dataList, prediction_method='decision_tree', saved_model_name='num_decision')
    train_classifier(data_list=dataList, prediction_method='knn', saved_model_name='num_knn')