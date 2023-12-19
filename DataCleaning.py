import pandas as pd
from Functions import outlier_detection
pd.set_option('display.max_columns', None)

df = pd.read_csv("online_retail.csv", parse_dates=['InvoiceDate'])
data = df.copy()

# cleaning the data
"""
There are rows in InvoiceNo column with value of 'C536379' which are all cancelled transactions that we should drop.
There are 9288 cancelled transactions.
"""
data = data[~data['InvoiceNo'].str.contains("C")].reset_index(drop=True)

"""
customerID column has many null values; since in market basket analysis we do not need the customerID rather InvoiceNo
thus we can either drop or ignore the column.
there are 1454 null values in description column as well which contains the core data in this analysis, thus we drop
all rows containing null values in that column.
"""
data = data[~data['Description'].isna()].reset_index(drop=True)

"""
The min and max values for Quantity column are definitely wrong! (-80995, 80995)
further investigation to delete rows with invalid data
9762 rows with quantity of bellow zero which is illogical thus those rows should also be dropped.
There seems to be only one row with extremely large quantity (80995) which will be dropped in the process of deleting
outliers.
"""
data = data[data['Quantity'] > 0].reset_index(drop=True)

"""
The next issue is the minimum and maximum value for UnitPrice (-11062.06, 13541.33)
There are only two rows with UnitPrice of bellow zero which will be dropped.
There are very few rows with high prices which will also be deleted in outlier detection.
"""
data = data[data['UnitPrice'] > 0].reset_index(drop=True)

# sns.histplot(data['Quantity'], bins=5000)
# plt.xlim([0, 1000])
# plt.show()

# outlier detection
"""
We are only interested in deleting high z-score values since the data is extremely right skewed.
even with z-score of 1.96 the data is still extremely skewed but choosing a z-score lower than that is unreasonable.
"""
columns = ['Quantity', 'UnitPrice']
data = outlier_detection(df=data, columns=columns, threshold=1.96).reset_index(drop=True)

"""
The unique values of the two columns StockCode and Description are 3913 and 4016 respectively which shows that
there are wrong values in dataset, since each item should only be associated with a unique StockCode and vice versa.
deleting the wrong values should be done in two steps:
"""
# step 1:
data_product = data[["Description", "StockCode"]].drop_duplicates()
data_product = data_product.groupby(["Description"]).agg({"StockCode": "count"}).reset_index()
data_product = data_product[data_product["StockCode"] > 1]
# There are 131 rows items StockCode value of greater than 1 which should all be dropped.
data = data[~data["Description"].isin(data_product["Description"])]

# step 2:
data_product = data[["Description", "StockCode"]].drop_duplicates()
data_product = data_product.groupby(["StockCode"]).agg({"Description": "count"}).reset_index()
data_product = data_product[data_product["Description"] > 1]
# All the rows with Description value of greater than 1 which should all be dropped.
data = data[~data["StockCode"].isin(data_product["StockCode"])]
# now both StockCode and Description have equally 3459 unique values.

# export the cleaned dataset
data.to_csv("./cleaned_retail_data.csv")
