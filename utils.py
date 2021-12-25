import joblib
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def normalize_data(data):
    """
    normalizes numerical data
    :param data: type = list, value list
    :return: normalized data list
    """
    scaler = MinMaxScaler()
    new_data = scaler.fit_transform(data)
    return new_data


def drop_outlier(data):
    """
    removes outlier from data
    :param data: pandas dataframe that contains ONLY numerical values
    :return: pandas dataframe with removed outliers
    """
    data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
    return data


def save_model(model, model_name):
    """
    Save the desired model with its name before prediction.
    Optional for classification method. Model saving takes place in prediction method.
    :param model: classifier model
    :param model_name: given classifier model name
    :return:
    """
    joblib.dump(model, 'models/' + str(model_name) + '.sav')


def load_model(model_name):
    """
    Loading saved model before prediction.
    Optional for classification method. Model loading takes place in prediction method.
    :param model_name: desired model name to load.
    :return:
    """
    loaded_model = joblib.load(model_name)
    print(f'{model_name} model loaded successfully !')

    return loaded_model


def timeit(start):
    """
    Find time between end time and given start time.
    :param start: Given start time for any operation.
    :return: datetime
    """
    stop = datetime.now()
    return stop - start


def hot(x):
    low = x.lower()
    if 'fair' in low:
        return 0
    elif 'rainy' in low:
        return 1
    elif 'snow' in low:
        return 2
    else:
        return 99
