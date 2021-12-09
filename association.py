from pandas.core.indexes.base import ensure_index
from pandas.io.parsers import read_csv
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
from fpgrowth_py import fpgrowth
import numpy as np


data = pd.read_csv('update.csv')


# smaller support value with <25
def hot_encode_severity(x):
    if isinstance(x, int) and x >= 3:
        return 1
    if isinstance(x, str) and x == "Day":
        return 1
    if isinstance(x, float) and x >= 25.0:
        return 1
    return 0


def hot_encode_visibility(x):
    if isinstance(x, str):
        return x == "Fair"
    return x >= 9.50


def hot_encode_distance(x):
    if isinstance(x, int):
        return x < 3
    return x <= 1.0


def restruct(data, columns):
    # print(data.columns)
    for col in columns:
        # print('col:', col)
        data = data[col]
    return data

# def apply_severity(file):
#     global data
#     f = open('csv.txt', mode='w')
#     # records = csv_to_list(file)
#     # data = pd.read_csv(file)
#     # data = data.head(100)
#     # association = apriori(records, min_support=0.6)
#     # night = (data[data['Sunrise_Sunset'] == "Night"]
#     #          .groupby(['Severity'])['ID']).count()
#     # traffic impacts with mean of temperature for accidents happened at night
#     # night = (data[data['Sunrise_Sunset'] == 'Night']
#     #          .groupby(['Severity'])['Temperature(F)']
#     #          .mean().reset_index()
#     #          .set_index('Severity'))

#     # print(night)
#     data = data[["Temperature(F)", "Sunrise_Sunset", "Severity"]]
#     # data_subset = data[(data["Temperature(F)"] >= 30.0)]

#     data_subset = data.applymap(hot_encode_severity)
#     data = data_subset
#     frq_items = apriori(data, min_support=0.004, use_colnames=True)

#     # Collecting the inferred rules in a dataframe
#     rules = association_rules(frq_items, metric="lift")
#     print(rules, len(rules))
#     f.write(rules)
#     rules = rules.sort_values(['confidence', 'lift'], ascending=[True, True])
#     print(rules.head())
#     f.close()
#     # association_results = list(association)
#     # print(association[0])


# def apply_visibility(file):
#     global data
#     # data = pd.read_csv(file)
#     data = data[["Visibility(mi)", "Weather_Condition"]]
#     data_subset = data.applymap(hot_encode_visibility)
#     data = data_subset
#     frq_items = apriori(data, min_support=0.004, use_colnames=True)

#     # Collecting the inferred rules in a dataframe
#     rules = association_rules(frq_items, metric="lift")
#     print(rules, len(rules))
#     rules = rules.sort_values(['confidence', 'lift'], ascending=[True, True])
#     print(rules.head())


# def apply_distance():
#     global data
#     data = data[["Distance(mi)", "Severity"]]
#     data_subset = data.applymap(hot_encode_distance)
#     data = data_subset
#     frq_items = apriori(data, min_support=0.0045, use_colnames=True)

#     # Collecting the inferred rules in a dataframe
#     rules = association_rules(frq_items, metric="lift")
#     print(rules, len(rules))
#     rules = rules.sort_values(['confidence', 'lift'], ascending=[True, True])
#     print(rules.head())


def apply_apriori(encode, columns, min_support):
    global data
    data = data[[c for c in data.columns if c in columns]]
    possibles = globals().copy()
    possibles.update(locals())
    method = possibles.get(encode)
    data_subset = data.applymap(method)
    data = data_subset
    freq_item_set = apriori(data, min_support=min_support, use_colnames=True)

    # Collecting the inferred rules in a dataframe
    rules = association_rules(freq_item_set, metric="lift", min_threshold=0.7)
    print(rules, len(rules))
    rules = rules.sort_values(['confidence', 'lift'], ascending=[True, True])
    print(rules.head())


def apply_fpgrowth(columns):
    global data
    data = data[[c for c in data.columns if c in columns]]
    # data = np.vstack(data)
    item_list = data.to_numpy()
    freq_item_set, _ = fpgrowth(item_list, minSupRatio=0.02, minConf=0.5)
    # patterns = pyfpgrowth.find_frequent_patterns(data, 0.02)
    # rules = pyfpgrowth.generate_association_rules(patterns, 0.5)
    # print("Rules: ", rules)
    rules = association_rules(freq_item_set, metric="confidence", min_threshold=0.7)
    print(rules, len(rules))
    rules = rules.sort_values(['confidence', 'lift'], ascending=[True, True])
    print(rules.head())


def fpgrowth2(columns):
    global data
    data = data[[c for c in data.columns if c in columns]]
    data = data.applymap(hot_encode_severity)
    # data_subset = data.applymap(hot_encode_severity)
    # data = np.vstack(data)
    # item_list = data.to_numpy()
    # transactions = data_subset
    # my_transactionencoder = TransactionEncoder()
    # fit the transaction encoder using the list of transaction tuples
    # my_transactionencoder.fit(transactions)
    print(data, data.columns)
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    print(te_ary)
    df = pd.DataFrame(te_ary, columns=columns)
    print(df)
    exit(3)
    # transform the list of transaction tuples into an array of encoded transactions
    encoded_transactions = my_transactionencoder.transform(transactions)

    # convert the array of encoded transactions into a dataframe
    encoded_transactions_df = pd.DataFrame(encoded_transactions, columns=my_transactionencoder.columns_)
    min_support = 7/len(transactions) 

    # compute the frequent itemsets using fpgriowth from mlxtend

    frequent_itemsets = fpgrowth(encoded_transactions_df, minSupRatio=min_support, minConf=0.5)

    # print the frequent itemsets
    # frequent_itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    print(rules)


if __name__ == '__main__':
    # apply_severity()
    # apply_visibility()
    # apply_distance()
    # exit(5)
    metric = input('->')
    f = open('data/config.json', mode='r')
   
    json_data = json.load(f)
    metric_dict = json_data[metric]
    encode = metric_dict['encode']
    columns = metric_dict['columns']
    min_support = metric_dict['min_support']

    # apply_apriori(encode, columns, min_support)
    # apply_fpgrowth(columns=columns)
    fpgrowth2(columns)