import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)

df = pd.read_csv("cleaned_retail_data.csv", parse_dates=['InvoiceDate'])
data = df.copy()

# since the data is too big and finding associations will get a huge amount of space, we will do it per country
data = data[data['Country'] == 'Germany']
# data.to_csv("Germany_clean_data.csv")

# creating a list of lists of items in dataframe
grouped_data = data['Description'].groupby(data['InvoiceNo']).apply(list)
data_items = grouped_data.values.tolist()

encoder = TransactionEncoder().fit(data_items)
transformed_encoder = encoder.transform(data_items)
encoded_data = pd.DataFrame(transformed_encoder, columns=encoder.columns_)


# applying apriori algorithm to find useful associations
frequent_items = apriori(encoded_data, min_support=0.01, max_len=3, use_colnames=True)
rules = association_rules(frequent_items, metric='support', min_threshold=0)
rules['antecedents'] = rules['antecedents'].apply(lambda a: ",".join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ",".join(list(a)))
# rules.to_csv("rules.csv")


# visualizing the boundaries by scatterplot to determine the thresholds of the rule
# sns.scatterplot(data=rules, x="antecedent support", y="consequent support", hue="lift", size="confidence")
# plt.show()
# the plot shows that antecedent support < 0.17 and consequent support < 0.17 gives the best rules thus:

filtered_rules = rules[(rules['antecedent support'] > 0.1) & (rules['support'] > 0.05) &
                       (rules['confidence'] > 0.2) & (rules['zhangs_metric'] > 0.6)]
# 40 useful rules are detected we cn work on

# visualizing the rules

# HEATMAP
support_table = filtered_rules.pivot(index='consequents', columns='antecedents', values='confidence')
sns.heatmap(data=support_table, cmap='viridis', annot=True, cbar=False)
plt.show()

"""
Based on the result of the heatmap we can perfectly see which of the items are correlated to use the acquired knowledge
to improve the sale and profit.
"""
