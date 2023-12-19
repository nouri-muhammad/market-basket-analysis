import numpy as np
import pandas as pd


def outlier_detection(df: pd.DataFrame, columns: list, threshold: float):
    """
    This function takes a dataframe, columns we need delete outliers from and a threshold for z-score then
    drops any rows from data that its value is above the threshold
    """
    for column in columns:
        df['z_score'] = (df[f"{column}"] - df[f"{column}"].mean()) / df[f"{column}"].std()
        df = df[df['z_score'] < threshold]
        df = df.drop(columns='z_score', axis=1)
    return df


def product_id_input(dataframe):
    """
    This function takes an id and check if an item with that id exists, if yes then returns the id otherwise asks
    for another id.
    """
    while True:
        product_id = str(input("product id: "))
        if str(product_id) in dataframe['StockCode'].values:
            return product_id
        else:
            print("no item with this id exists!")


def check_item_by_id(dataframe, stock_code):
    """
    This function accepts an item id and returns the item name.
    """
    product_name = dataframe[dataframe["StockCode"] == stock_code]["Description"].unique()[0]
    return product_name


def product_antecedents(dataframe, product_name):
    """
    This function takes a dataframe(rule dataframe) with the name of an item and returns a dataframe including
    all the rows in which that item exist in antecedents column.
    """
    df = dataframe[dataframe['antecedents'].str.contains(product_name)]
    return df


def recommendation_items(dataframe):
    """
    This function takes the dataframe of product_antecedents function.
    if a consequent item has "consequent support", "lift" and "zhangs_metric" of more than 40% quantile,
    then maximum 3 of those items should be recommended based on zhangs_metric in descending order.
    """
    consequent_q = dataframe['consequent support'].quantile(0.4)
    lift_q = dataframe['lift'].quantile(0.4)
    zhangs_metric_q = dataframe['zhangs_metric'].quantile(0.4)
    recommendations = dataframe[(dataframe['consequent support'] > consequent_q) &
                                (dataframe['lift'] > lift_q) &
                                (dataframe['zhangs_metric'] > zhangs_metric_q)]
    recommendations = recommendations.sort_values('zhangs_metric', ascending=False)
    return recommendations[:3]['consequents'].unique()


# ############################################## NOTE ################################################### #
# The following functions were created in case of need in special circumstances and for educational purposes.


def support(df: pd.DataFrame, x: str, y: str):
    """
    This function calculates individual support values for any column
    keep in mind that the support of not happening of a column is (1-happening)
    """
    dataframe = df.copy()
    support_x = dataframe[f"{x}"].mean()
    support_y = dataframe[f"{y}"].mean()
    return support_x, support_y


def combination_support(df: pd.DataFrame, x: str, y: str):
    """
    This function calculates the support value for combination of two columns
    """
    dataframe = df.copy()
    dataframe[f"{x} + {y}"] = np.logical_and(dataframe[f"{x}"], dataframe[f"{y}"])
    support_xy = dataframe[f"{x} + {y}"].mean()
    return support_xy


def negative_combo_support(df: pd.DataFrame, x: str, y: str):
    """
    This function returns the support of when x happens and y does not happen
    """
    dataframe = df.copy()
    dataframe[f"{x} not {y}"] = np.logical_and(dataframe[f"{x}"], ~dataframe[f"{y}"])
    support_xny = dataframe[f"{x} not {y}"].mean()
    return support_xny


def confidence(df: pd.DataFrame, x: str, y: str):
    """
    This function calculates the confidence value for x -> y and return it.
    """
    support_xy = combination_support(df, x, y)
    support_x, support_y = support(df, x, y)
    confidence_value = support_xy / support_x
    return confidence_value


def lift(df: pd.DataFrame, x: str, y: str):
    """
    This function calculates the lift value for x -> y and return it.
    """
    support_xy = combination_support(df, x, y)
    support_x, support_y = support(df, x, y)
    lift_value = support_xy / (support_x * support_y)
    return lift_value


def leverage(df, x, y):
    """
    This function calculates the leverage value for x -> y and return it.
    """
    support_xy = combination_support(df, x, y)
    support_x, support_y = support(df, x, y)
    leverage_value = support_xy - (support_x * support_y)
    return leverage_value


def conviction(df, x, y):
    """
    This function calculates the conviction value for x -> y and return it.
    """
    support_xny = negative_combo_support(df, x, y)
    support_x, support_y = support(df, x, y)
    conviction_value = (support_x * (1 - support_y)) / support_xny
    return conviction_value


def zhang(df, x, y):
    """
    This function calculates the zhang value for x -> y and return it.
    """
    support_x, support_y = support(df, x, y)
    support_xy = combination_support(df, x, y)
    zhang_value = (support_xy - (support_x * support_y)) / max(support_xy * (1-support_x),
                                                               support_x * (support_y-support_xy))
    return zhang_value
