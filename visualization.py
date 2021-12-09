import pydotplus
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pandas as pd
import numpy as np
import seaborn
import plotly.express as px

dataset = pd.read_csv('update.csv')

'''
For decision tree png version.
'''
def pydot_visual(dtree, features):
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')

    img=pltimg.imread('mydecisiontree.png')
    plt.imshow(img)
    # plt.show()


def matplotlib_visual(columns):
    data = dataset[[c for c in dataset.columns if c in columns]]
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
    data = dataset[[c for c in dataset.columns if c in columns]]
    seaborn.set(style='whitegrid')
    # fmri = seaborn.load_dataset("fmri")

    seaborn.scatterplot(x="Severity",
                        y="Visibility(mi)",
                        data=data)
    plt.show()


def plotly_visual(column, name):
    # plotting the pie chart
    # data = dataset[[c for c in dataset.columns if c in columns]]
    fig = px.pie(dataset, values=column, names=name)
    
    # showing the plot
    fig.show()

if __name__ == '__main__':
    # matplotlib_visual('Timezone')
    # print(dataset['Visibility(mi)'].max())
    # seaborn_visual(['Visibility(mi)', 'Severity'])
    plotly_visual('Severity', 'Severity')
