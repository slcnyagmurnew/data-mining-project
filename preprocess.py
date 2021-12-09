import numpy as np
import pandas as pd

predict_list = ['Severity', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                'Wind_Speed(mph)', 'Precipitation(in)']

column_list = ['ID', 'Severity', 'Start_Time', 'Distance(mi)', 'Street', 'City', 'Timezone', 'Temperature(F)',
               'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Precipitation(in)', 'Weather_Condition', 'Junction',
               'Railway', 'Stop', 'Traffic_Signal', 'Sunrise_Sunset']


def calculate_rowsize(input_file, file_count):
    """
    This method calculates number of rows that should be in splitted csv files according to number of sub files.
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


# ID,Severity,Start_Time,End_Time,Start_Lat,Start_Lng,End_Lat,End_Lng,Distance(mi),Description,Number,Street,Side,City,County,State,Zipcode,Country,Timezone,Airport_Code,Weather_Timestamp,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Direction,Wind_Speed(mph),Precipitation(in),Weather_Condition,Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop,Sunrise_Sunset,Civil_Twilight,Nautical_Twilight,Astronomical_Twilight

# split_csv('data\\US_Accidents_Dec20_updated.csv', 20, 'data_parts\\')

# df = pd.read_csv('data\\US_Accidents_Dec20_updated.csv')
# get_statistics(dataframe=df, s_type='all')


def get_statistics(dataframe, s_type='mean'):
    for i in range(len(dataframe.columns)):
        if type(dataframe.iloc[0, i]) in [np.int64, np.int, np.int32, np.float64, np.float32, float, int]:
            if s_type is 'mean':
                print(f'Column Name : {dataframe.columns[i]}, Mean = {dataframe.iloc[:, i].mean()}')
            elif s_type is 'mode':
                print(f'Column Name : {dataframe.columns[i]}, Mode = {dataframe.iloc[:, i].mode()[0]}')
            elif s_type is 'median':
                print(f'Column Name : {dataframe.columns[i]}, Median = {dataframe.iloc[:, i].median()}')
            elif s_type is 'all':
                print(f'Column Name : {dataframe.columns[i]}, Mean = {dataframe.iloc[:, i].mean()},'
                      f' Mode = {dataframe.iloc[:, i].mode()[0]}, Median = {dataframe.iloc[:, i].median()}')
            else:
                print('Please give correct s_type!')
                pass
        elif type(dataframe.iloc[0, i]) in [bool, np.bool, np.bool_]:
            print(f'Column Name : {dataframe.columns[i]}, Boolean Count = {dataframe.iloc[:, i].value_counts()}')


def split_csv_by_column_name(input_file, out_dir, column_names):
    df = pd.read_csv(input_file)
    print('Input csv read !')
    new_df = pd.DataFrame()

    for c_name in column_names:
        new_df[c_name] = df[c_name]

    new_df.dropna().to_csv(path_or_buf=out_dir, index=False)
    print('Out csv written !')
    pass


def split_csv_by_data_type(input_file, out_dir, data_types, classifier_class_name=None):
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


# split_csv_by_data_type(input_file='data\\US_Accidents_Dec20_updated.csv', out_dir='splitted_data\\1.csv',
#                        data_types=[np.int64, np.int, np.int32, np.float64, np.float32, float, int])
if __name__ == "__main__":
    df = pd.read_csv('data/US_Accidents_Dec20_updated.csv')
    split_csv_by_column_name('data/US_Accidents_Dec20_updated.csv', 'update.csv', column_names=column_list)
    # print(df["Weather_Condition"].unique())
    # split_csv_by_column_name(input_file='data\\US_Accidents_Dec20_updated.csv', out_dir='splitted_data\\1.csv',
    #                         column_names=predict_list)
