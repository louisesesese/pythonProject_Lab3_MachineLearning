import pandas as pd
from decimal import Decimal
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv('dataset_group.csv',header=None)

unique_id = list(set(data[1]))
num_unique_id = len(unique_id)
print(num_unique_id)

items = list(set(data[2]))
num_items = len(items)
print(num_items)

dataset = [[elem for elem in data[data[1] == id][2] if elem in items] for id in unique_id]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
print(results)

results1 = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
print(results1)

results2 = apriori(df, min_support=0.3, use_colnames=True)
results2['length'] = results2['itemsets'].apply(lambda x: len(x))
results2 = results2[results2['length'] == 2]
print(results2)
print('\nCount of result itemstes = ', len(results2))

start_sup = Decimal(0.05)
support = []
length = []
max_len = []
while start_sup <= 1:
    support.append(start_sup)
    results = apriori(df, min_support=start_sup, use_colnames=True)
    results['length'] = results['itemsets'].apply(lambda x: len(x))

    length.append(len(results))
    if len(results) > 0:
        max_len.append(max(results['length']))

    start_sup += Decimal(0.01)

fig1, ax = plt.subplots(figsize=(6, 4))
ax.plot(support, length)

ax.scatter(support[max_len.index(3)], length[max_len.index(3)], color='blue', s=40, marker='o')
ax.scatter(support[max_len.index(2)], length[max_len.index(2)], color='purple', s=40, marker='o')
ax.scatter(support[max_len.index(1)], length[max_len.index(1)], color='green', s=40, marker='o')
ax.scatter(support[len(max_len)], length[len(max_len)], color='yellow', s=40, marker='o')

plt.show()

results3 = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results3['itemsets']]
new_dataset = [[elem for elem in data[data[1] == id][2] if elem in new_items] for id in unique_id]

te1 = TransactionEncoder()
te_ary1 = te1.fit(new_dataset).transform(new_dataset)
df1 = pd.DataFrame(te_ary1, columns=te1.columns_)

new_results1 = apriori(df1, min_support=0.3, use_colnames=True)
association_rules(new_results1, min_threshold=0.3)

new_results2 = apriori(df1, min_support=0.15, use_colnames=True)
rules_2 = association_rules(new_results2, min_threshold=0.15)
rules_2["antecedent_len"] = rules_2["antecedents"].apply(lambda x: len(x))
rules_2[(rules_2['antecedent_len'] > 1) & (rules_2["antecedents"].apply(lambda x: 'yogurt' in str(x) or 'waffles' in str(x)))]

new_dataset_3 = [[elem for elem in data[data[1] == id][2] if elem not in new_items] for id in unique_id]

te3 = TransactionEncoder()
te_ary3 = te3.fit(new_dataset_3).transform(new_dataset_3)
df3 = pd.DataFrame(te_ary3, columns=te3.columns_)

results_3 = apriori(df3, min_support=0.05, use_colnames=True, max_len=10)
def check_first_s(string_list: list) -> bool:
    return len(list(filter(lambda y: y[0] == 's', string_list))) > 1

rules_3 = association_rules(results_3, min_threshold=0.05)
rules_3["antecedent_len"] = rules_3["antecedents"].apply(lambda x: len(x))
rules_3[(rules_3['antecedent_len'] > 1) & (rules_3["antecedents"].apply(lambda x: check_first_s(list(x))))]
rules_3[(rules_3['support'] >= 0.1) & (rules_3["support"] <= 0.25)]