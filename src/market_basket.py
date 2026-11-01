"""Market-basket analysis of the Kaggle "Groceries dataset" with Apriori.

Reconstructed from the code listings in the project report (originally run in
Google Colab, Nov 2025) and tidied into one script. The only step not shown in
the report is how `daily_counts` was built; it is inferred below as the number
of distinct transactions (member + date baskets) per day.

Usage:
    python src/market_basket.py --data Groceries_dataset.csv --out figures/
Dataset: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from mlxtend.frequent_patterns import apriori, association_rules  # noqa: E402
from mlxtend.preprocessing import TransactionEncoder  # noqa: E402

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def save(out_dir, name):
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=150)
    plt.close()


def load(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    return df


def eda(df, out_dir):
    top_20 = df["itemDescription"].value_counts().head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_20.values, y=top_20.index, palette="viridis")
    plt.title("Top 20 most frequently purchased items")
    save(out_dir, "top20_items.png")

    # Visits per customer = distinct shopping dates (not item rows)
    visits = df.groupby("Member_number")["Date"].nunique()
    plt.figure(figsize=(10, 6))
    sns.histplot(visits, bins=30, color="green")
    plt.title("Transactions (visits) per customer")
    save(out_dir, "visits_per_customer.png")
    return visits


def time_series(df, out_dir):
    # Inferred: one transaction = one member's basket on one date
    daily_counts = df.groupby("Date")["Member_number"].nunique()

    monthly = daily_counts.resample("ME").sum()   # "M" in older pandas
    monthly.index = monthly.index.strftime("%Y-%m")
    plt.figure(figsize=(15, 7))
    sns.barplot(x=monthly.index, y=monthly.values, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Total monthly transactions (2014-2015)")
    save(out_dir, "monthly_transactions.png")

    weekday = daily_counts.groupby(daily_counts.index.day_name()).mean().reindex(DAY_ORDER)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=weekday.index, y=weekday.values, order=DAY_ORDER, palette="icefire")
    plt.title("Mean transactions by day of week")
    save(out_dir, "weekday_mean.png")
    return daily_counts


def build_baskets(df):
    baskets = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).tolist()
    te = TransactionEncoder()
    encoded = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
    print(f"Matrix shape: {encoded.shape} (transactions, items)")   # report: (14963, 167)
    return encoded


def support_elbow(encoded, out_dir):
    thresholds = np.arange(0.001, 0.05, 0.002)
    counts = [len(apriori(encoded, min_support=s, use_colnames=True)) for s in thresholds]
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, counts, marker="o")
    plt.xlabel("Minimum support threshold")
    plt.ylabel("Number of frequent itemsets")
    plt.title("Selecting min_support (elbow)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    save(out_dir, "support_elbow.png")


def mine_rules(encoded, min_support=0.01, min_confidence=0.1):
    itemsets = apriori(encoded, min_support=min_support, use_colnames=True)
    itemsets = itemsets.sort_values("support", ascending=False)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    print(f"Frequent itemsets: {len(itemsets)} | rules: {len(rules)}")
    # Lift < 1 means the items co-occur LESS often than independence predicts
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
          .sort_values("lift", ascending=False).to_string())
    return itemsets, rules


def rules_network(rules, out_dir, top_n=15):
    top = rules.sort_values("lift", ascending=False).head(top_n).copy()
    top["a"] = top["antecedents"].apply(lambda x: ", ".join(x))
    top["c"] = top["consequents"].apply(lambda x: ", ".join(x))
    g = nx.from_pandas_edgelist(top, source="a", target="c", edge_attr=["confidence", "lift"])
    pos = nx.spring_layout(g, k=0.8, seed=42)
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(g, pos, node_size=3000, node_color="lightblue", alpha=0.8)
    nx.draw_networkx_labels(g, pos, font_size=10)
    nx.draw_networkx_edges(g, pos, width=top["confidence"] * 5, edge_color="grey", alpha=0.6)
    labels = {(r["a"], r["c"]): f"Lift: {r['lift']:.2f}" for _, r in top.iterrows()}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_size=8)
    plt.axis("off")
    plt.title(f"Top {min(top_n, len(top))} rules by lift")
    save(out_dir, "rules_network.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Groceries_dataset.csv")
    p.add_argument("--out", default="figures")
    p.add_argument("--min-support", type=float, default=0.01)
    p.add_argument("--min-confidence", type=float, default=0.1)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.data)
    print(f"Rows: {len(df):,}")                                      # report: 38,765
    eda(df, args.out)
    time_series(df, args.out)
    encoded = build_baskets(df)
    support_elbow(encoded, args.out)
    _, rules = mine_rules(encoded, args.min_support, args.min_confidence)
    if len(rules):
        rules_network(rules, args.out)


if __name__ == "__main__":
    main()
