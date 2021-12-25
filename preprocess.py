import numpy as np
import pandas as pd
from sklearn.utils import shuffle

input_file = '/home/yagmur/Desktop/US_Accidents_Dec20_updated.csv'

predict_list = ['Severity', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                'Wind_Speed(mph)', 'Precipitation(in)']

column_list = ['Severity', 'Distance(mi)', 'City', 'Timezone', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
               'Crossing', 'Wind_Speed(mph)', 'Wind_Chill(F)', 'Visibility(mi)', 'Precipitation(in)',
               'Weather_Condition', 'Junction', 'Railway', 'Stop', 'Traffic_Signal', 'Sunrise_Sunset']


def calculate_rowsize(input_file, file_count):
    """
    This method calculates number of rows that should be in split csv files according to number of sub files.
    :param input_file: name of input csv file with path
    :param file_count: number of sub files
    :return: number of rows
    """
    number_lines = sum(1 for row in (open(input_file)))
    div = number_lines / file_count
    mod = number_lines % file_count
    print(f'Div = {div}, Mod = {mod}')
    return div


def split_csv(input_file, file_count, out_dir):
    """
    This method splits main csv file into sub csv files.
    :param input_file: name of input csv file with path
    :param file_count: number of sub files (real number of sub files can be +1)
    :param out_dir: dir of output csv files
    """
    number_lines = sum(1 for row in (open(input_file)))
    row_size = calculate_rowsize(input_file=input_file, file_count=file_count)
    row_size = int(row_size)
    # start looping through data writing it to a new file for each set
    for i in range(1, number_lines, row_size):
        df = pd.read_csv(input_file,
                         header=None,
                         nrows=row_size,  # number of rows to read at each loop
                         skiprows=i)  # skip rows that have been read

        # csv to write data to a new file with indexed name. input_1.csv etc.
        out_csv = out_dir + str(i) + '.csv'
        df.to_csv(out_csv,
                  index=False,
                  header=False,
                  mode='a',  # append data to csv file
                  chunksize=row_size)  # size of data to append for each loop
    pass


def get_statistics(dataframe, s_type='mean'):
    """
    This method calculates statistics of dataframe for numeric columns.
    :param dataframe: Input dataframe
    :param s_type: Statistic type. It can be mean, median, mode or all to calculate all of them.
    """
    for i in range(len(dataframe.columns)):
        if type(dataframe.iloc[0, i]) in [np.int64, np.int, np.int32, np.float64, np.float32, float, int]:
            if s_type == 'mean':
                print(f'Column Name : {dataframe.columns[i]}, Mean = {dataframe.iloc[:, i].mean()}')
            elif s_type == 'mode':
                print(f'Column Name : {dataframe.columns[i]}, Mode = {dataframe.iloc[:, i].mode()[0]}')
            elif s_type == 'median':
                print(f'Column Name : {dataframe.columns[i]}, Median = {dataframe.iloc[:, i].median()}')
            elif s_type == 'all':
                print(f'Column Name : {dataframe.columns[i]}, Mean = {dataframe.iloc[:, i].mean()},'
                      f' Mode = {dataframe.iloc[:, i].mode()[0]}, Median = {dataframe.iloc[:, i].median()}')
            else:
                print('Please give correct s_type!')
                pass
        elif type(dataframe.iloc[0, i]) in [bool, np.bool, np.bool_]:
            print(f'Column Name : {dataframe.columns[i]}, Boolean Count = {dataframe.iloc[:, i].value_counts()}')


def split_csv_by_column_name(input_file, out_dir, column_names):
    """
    This method splits input csv by given column names, drops NA values and saves output csv file.
    :param input_file: Input file name that will be split with directory name.
    :param out_dir: Directory of output csv file with csv name.
    :param column_names: Column names that will be used for split.
    """
    df = pd.read_csv(input_file)
    print('Input csv read !')
    new_df = pd.DataFrame()

    for c_name in column_names:
        new_df[c_name] = df[c_name]

    new_df.dropna().to_csv(path_or_buf=out_dir, index=False)
    print('Out csv written !')
    pass


def split_csv_by_data_type(input_file, out_dir, data_types, classifier_class_name=None):
    """
    This method splits input csv by given data types, drops NA values and saves output csv file.
    :param input_file: Input file name that will be splitted with directory name.
    :param out_dir: Directory of output csv file with csv name.
    :param data_types: Data types that will be used for split.
    :param classifier_class_name: Classifier column name. If it is not None, a column that with name
     classifier_class_name will be added to output csv.
    """
    df = pd.read_csv(input_file)
    print('Input csv read !')
    new_df = pd.DataFrame()

    for i in range(len(df.columns)):
        if type(df.iloc[0, i]) in data_types:
            if df.columns[i] != classifier_class_name:
                new_df[df.columns[i]] = df.iloc[:, i]

    if classifier_class_name is not None:
        new_df[classifier_class_name] = df[classifier_class_name]

    new_df.dropna().to_csv(path_or_buf=out_dir, index=False)
    print('Out csv written !')
    pass


def split_to_train(input_file, out_dir, size):
    """
    This method splits input csv by 'Severity' column, drops NA values and saves output csv file.
    :param input_file: Input file name that will be split with directory name.
    :param out_dir: Directory of output csv file with csv name.
    :param size: Size of each row groups that has same 'Severity' value.
    """
    counts = [0, 0, 0, 0]
    df = pd.read_csv(input_file)
    df = df.dropna()
    df = shuffle(df)
    print('Input csv read !')
    new_df = pd.DataFrame()

    for i in range(len(df)):
        if counts[int(df.iloc[i]['Severity']) - 1] < size:
            new_df = new_df.append(df.iloc[i], ignore_index=True)
            counts[int(df.iloc[i]['Severity']) - 1] += 1
        sum_counts = counts[0] + counts[1] + counts[2] + counts[3]
        if sum_counts >= size * len(counts):
            break
    new_df.to_csv(path_or_buf=out_dir, index=False)
    print('Out csv written !')
    pass


if __name__ == '__main__':
    # ID,Severity,Start_Time,End_Time,Start_Lat,Start_Lng,End_Lat,End_Lng,Distance(mi),Description,Number,Street,Side,City,County,State,Zipcode,Country,Timezone,Airport_Code,Weather_Timestamp,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Direction,Wind_Speed(mph),Precipitation(in),Weather_Condition,Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop,Sunrise_Sunset,Civil_Twilight,Nautical_Twilight,Astronomical_Twilight

    # split_csv('data\\US_Accidents_Dec20_updated.csv', 20, 'data_parts\\')
    df = pd.read_csv('data/split_last.csv')
    print(df['Severity'].value_counts())
    # get_statistics(dataframe=df, s_type='all')
    # split_to_train('data/last.csv', 'data/split_last.csv', 3000)
    # split_csv_by_data_type(input_file='data/US_Accidents_Dec20_updated.csv', out_dir='splitted_data/INT.csv',
    #                        data_types=[np.int64, np.int, np.int32, np.float64, np.float32, float, int])

    # split_csv_by_column_name(input_file='data\\US_Accidents_Dec20_updated.csv', out_dir='splitted_data\\1.csv',
    #                          column_names=predict_list)

    # split_to_train(input_file=input_file, out_dir='data/splitted_data.csv', size=3000)
    # split_csv_by_column_name(input_file='/home/yagmur/Desktop/US_Accidents_Dec20_updated.csv',
    #                          out_dir='data/last.csv', column_names=column_list)
    # df = pd.read_csv('data/last.csv')
    # print(len(df.columns))
    # my_list = ['Severity', 'Junction', 'Traffic_Signal', 'Crossing', 'Weather_Condition', 'Sunrise_Sunset']
    # split_csv_by_column_name(input_file='data/US_Accidents_Dec20_updated.csv', out_dir='my.csv', column_names=my_list)
    # df = pd.read_csv('my.csv')
    # print(df['Weather_Condition'].unique())
    # df['Weather_Condition'] = df['Weather_Condition'].map(hot)
    # df['Sunrise_Sunset'] = df['Sunrise_Sunset'].map({'Day': 0, 'Night': 1})
    # df['Junction'] = df['Junction'].map({False: 0, True: 1})
    # df['Crossing'] = df['Crossing'].map({False: 0, True: 1})
    # df['Traffic_Signal'] = df['Traffic_Signal'].map({False: 0, True: 1})
    # df = df[df.Weather_Condition != 99]
    # df.to_csv('last.csv', index=False)

    # split_to_train(input_file='last.csv', out_dir='lt.csv', size=3000)

