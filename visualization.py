import re
import pydotplus
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pandas as pd
import numpy as np
import seaborn
import plotly.express as px
from preprocess import split_csv_by_data_type
from utils import drop_outlier

'''
This csv file was created by desired column values that split by specific number.
For example: Severity(1:3k data, 2:3k data, 3:3k data, 4:3k data)
'''

# dataset = pd.read_csv('data/splitted_data.csv')
dataset = pd.read_csv('data/last.csv')

numerical_columns = ['Severity', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Wind_Speed(mph)',
                     'Wind_Chill(F)', 'Visibility(mi)', 'Precipitation(in)', 'Start_Lat', 'End_Lat', 'End_Lng',
                     'Start_Lng']


def calc_percentage(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


def visualize_boolean_values(input_file, out_dir):
    dataset = pd.read_csv(input_file)
    print('CSV read !')

    for col in dataset:
        if type(dataset[col][1]) == bool or type(dataset[col][1]) == np.bool or type(dataset[col][1]) == np.bool_:

            try:
                l1 = dataset[col].value_counts().index.tolist()[0]
                v1 = dataset[col].value_counts()[0]

                try:
                    l2 = dataset[col].value_counts().index.tolist()[1]
                    v2 = dataset[col].value_counts()[1]

                except Exception as e:
                    v2 = 0
                    if l1 == 'True' or l1:
                        l2 = 'False'
                    else:
                        l2 = 'True'
                fig = plt.figure(figsize=(10, 7))
                plt.pie([v1, v2], labels=[l1, l2], autopct=lambda pct: calc_percentage(pct, [v1, v2]))
                plt.title(str(col))
                plt.savefig(out_dir + '/' + str(col) + '.png')

            except Exception as err:
                print(err)
                print(col)


def pydot_visual(dtree, features):
    """
    For decision tree png version.
    :param dtree: decision tree object; retrieved from classification function.
    :param features: list; decision tree classifier features
    :return: plt
    """
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')

    img = pltimg.imread('mydecisiontree.png')
    plt.imshow(img)
    # plt.show()


def matplotlib_visual(columns, method=None):
    """
    For timezone png version.
    If case and method is used to show mean, median and max values of numerical columns.
    :param columns: list; timezone column types in dataframe.
    :return: plt
    """
    data = dataset[[c for c in dataset.columns if c in columns]]

    if method is not None:
        col_list = []
        lst = []
        print(len(data.columns))
        for col in data.columns:
            res = re.sub(r"\((.*?)\)", "", col)
            val = None
            if method == 'Mean':
                val = data[col].mean()
            elif method == 'Median':
                val = data[col].median()
            elif method == 'Max':
                val = data[col].max()
            else:
                print('error')
            lst.append(val)
            col_list.append(res)
        print(len(lst))
        graph = {
            'Columns': col_list,
            f'{method}': lst
        }
        df = pd.DataFrame(graph)
        df.plot.bar(x='Columns', y=method, rot=0)
        plt.show()

    else:
        sf = data.value_counts()
        counts = np.array(sf.values.tolist())
        data = {'Counts': counts,
                'Timezone': ['US/Eastern',
                             'US/Pacific',
                             'US/Central',
                             'US/Mountain']}

        # Creates pandas DataFrame.
        df = pd.DataFrame(data)
        print(df, df.columns)
        df.plot.bar(x='Timezone', y='Counts', rot=0)
        plt.show()


def seaborn_visual(columns):
    """
    For severity-visibility association png version.
    :param columns: list; desired column names, in this example: Visibility and Severity.
    :return: plt
    """
    data = dataset[[c for c in dataset.columns if c in columns]]
    seaborn.set(style='whitegrid')
    # fmri = seaborn.load_dataset("fmri")

    seaborn.scatterplot(x="Severity",
                        y="Visibility(mi)",
                        data=data)
    plt.show()


def plotly_visual(column, name):
    """
    Pie chart visualization for severity column.
    :param column: string; desired column type.
    :param name: string; desired column name.
    :return: figure
    """
    # plotting the pie chart
    fig = px.pie(dataset, values=column, names=name)

    # showing the plot
    fig.show()


'''
severity.png and severity-drop.png were created by this function.
severity.png : normal severity values
severity-drop.png : with dropping outlier severity values
'''


def visualize_numeric_values(dataset, out_dir):
    """
    For severity.png and severity-drop.png files.
    severity.png : normal severity values
    severity-drop.png : with dropping outlier severity values
    :param dataset: dataframe; input csv file.
    :param out_dir: string; when wants to create csv file from dataframe.
    :return: plt
    """
    dataset = split_csv_by_data_type(df=dataset, out_dir=None,
                                     data_types=[np.int64, np.int, np.int32, np.float64, np.float32, float, int])
    dataset = drop_outlier(dataset)
    seaborn.pairplot(dataset)
    plt.savefig(out_dir + '.png')
    plt.show()


if __name__ == '__main__':
    matplotlib_visual(columns=numerical_columns, method='Max')
    # seaborn_visual(['Visibility(mi)', 'Severity'])
    # plotly_visual('Severity', 'Severity')
    # visualize_numeric_values(dataset=dataset, out_dir='data/severity-drop')
