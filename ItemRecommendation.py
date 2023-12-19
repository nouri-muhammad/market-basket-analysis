import pandas as pd
from Functions import check_item_by_id, product_id_input, product_antecedents, recommendation_items
pd.set_option('display.max_columns', None)

df = pd.read_csv("Germany_clean_data.csv", parse_dates=['InvoiceDate'])
data = df.copy()

rules = pd.read_csv("rules.csv")
data_rules = rules.copy()

# giving item recommendation for a chosen item (based on item id [StockCode]) for test use (22908, 22907, 22993, 22244)
product_id = product_id_input(dataframe=data)
product_name = check_item_by_id(dataframe=data, stock_code=product_id)
consequents = product_antecedents(data_rules, product_name)

# finding top items for recommendation from the consequents dataframe
df = recommendation_items(consequents)
print(df)
