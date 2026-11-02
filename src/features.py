import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Group"] = df["PassengerId"].str.split("_").str[0]

    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")

    df["GroupMember"] = df["PassengerId"].str.split("_").str[1]

    df["Group"] = pd.to_numeric(df["Group"], errors="coerce")

    df["GroupMember"] = pd.to_numeric(df["GroupMember"], errors="coerce")

    df["CabinDeck"] = df["Cabin"].str.split("/").str[0]
    # df["CabinDeck"] = df["CabinDeck"].astype("category")

    df["CabinNum"] = df["Cabin"].str.split("/").str[1]

    df["CabinSide"] = df["Cabin"].str.split("/").str[2]
    # df["CabinSide"] = df["CabinSide"].astype("category")

    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

    spending_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    df["TotalSpent"] = df[spending_columns].sum(axis=1, skipna=True)

    all_missing = df[spending_columns].isna().all(axis=1)

    df.loc[all_missing, "TotalSpent"] = np.nan

    df["ServicesUsed"] = (df[spending_columns] > 0).sum(axis=1)

    df["Surname"] = df["Name"].str.split().str[0]
    # df["Surname"] = df["Surname"].astype("category")

    df.drop(columns=["PassengerId", "Cabin", "Name"], inplace=True)

    return df


def categorize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # df["HomePlanet"] = df["HomePlanet"].astype("category")

    # df["CryoSleep"] = df["CryoSleep"].astype("bool")

    # df["Destination"] = df["Destination"].astype("category")

    # df["VIP"] = df["VIP"].astype("bool")

    if "Transported" in df.columns:
        df["Transported"] = df["Transported"].astype("bool")

    return df
