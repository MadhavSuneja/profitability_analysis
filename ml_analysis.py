import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

RAW_PATH = "Nassau Candy Distributor.csv"

# Create outputs directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 0 — Load & clean raw data
# ---------------------------------------------------------------------------
def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path)

    df["Sales"] = pd.to_numeric(
        df["Sales"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    )
    df["Cost"] = pd.to_numeric(
        df["Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    )

    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")

    df["Division"] = df["Division"].astype(str).str.strip().str.title()
    df["Product Name"] = df["Product Name"].astype(str).str.strip().str.title()
    df["Division"] = df["Division"].replace({
        "Confectionary": "Confectionery", "Confect": "Confectionery",
        "Snack": "Snacks", "Beverage": "Beverages"
    })
    df["Product Name"] = df["Product Name"].replace({
        "Choclate": "Chocolate", "Chococ Bar": "Chocolate Bar",
        "Candybox": "Candy Box", "CandyBox": "Candy Box"
    })

    df["Units"] = df["Units"].fillna(df["Units"].median())

    # keep only valid, sellable rows for modelling (mirrors Data_clean)
    df = df[(df["Sales"] > 0) & (df["Cost"] >= 0)].dropna(subset=["Sales", "Cost"]).copy()

    df["Profit"] = df["Sales"] - df["Cost"]
    df["Gross_Margin_%"] = (df["Profit"] / df["Sales"]) * 100
    df["Cost_to_Sales_%"] = (df["Cost"] / df["Sales"]) * 100
    df["Profit_per_Unit"] = df["Profit"] / df["Units"].replace(0, np.nan)
    return df


def build_product_table(df):
    prod = df.groupby("Product Name").agg(
        Division=("Division", "first"),
        Total_sales=("Sales", "sum"),
        Total_units=("Units", "sum"),
        Total_cost=("Cost", "sum"),
        Total_profit=("Profit", "sum"),
        Avg_Ship_Mode_Same_Day=("Ship Mode", lambda x: (x == "Same Day").mean()),
    ).reset_index()
    prod["Gross_Margin_%"] = prod["Total_profit"] / prod["Total_sales"] * 100
    prod["Profit_per_Unit"] = prod["Total_profit"] / prod["Total_units"]
    prod["Cost_to_Sales_%"] = prod["Total_cost"] / prod["Total_sales"] * 100
    prod["Revenue_Contribution_%"] = prod["Total_sales"] / prod["Total_sales"].sum() * 100
    prod["Profit_Contribution_%"] = prod["Total_profit"] / prod["Total_profit"].sum() * 100
    return prod


# ---------------------------------------------------------------------------
# 1) PRODUCT SEGMENTATION — KMeans
# ---------------------------------------------------------------------------
def segment_products(prod, k=4):
    features = ["Gross_Margin_%", "Total_sales", "Profit_per_Unit", "Revenue_Contribution_%"]
    X = prod[features].fillna(0)
    Xs = StandardScaler().fit_transform(X)

    # quick silhouette check across k=2..5 so k isn't picked blindly
    scores = {}
    for kk in range(2, min(6, len(prod))):
        labels = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(Xs)
        scores[kk] = silhouette_score(Xs, labels)
    best_k = max(scores, key=scores.get) if scores else k

    model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    prod = prod.copy()
    prod["Cluster"] = model.fit_predict(Xs)

    # label clusters by their mean margin & sales so output is human-readable
    summary = prod.groupby("Cluster")[["Gross_Margin_%", "Total_sales"]].mean()
    margin_mid = summary["Gross_Margin_%"].median()
    sales_mid = summary["Total_sales"].median()

    def label(row):
        high_margin = row["Gross_Margin_%"] >= margin_mid
        high_sales = row["Total_sales"] >= sales_mid
        if high_margin and high_sales:
            return "Star (high sales, high margin)"
        if high_sales and not high_margin:
            return "Cash Cow (high sales, low margin)"
        if not high_sales and high_margin:
            return "Niche (low sales, high margin)"
        return "Underperformer (low sales, low margin)"

    cluster_labels = summary.apply(label, axis=1)
    prod["Segment"] = prod["Cluster"].map(cluster_labels)
    return prod, scores, best_k


# ---------------------------------------------------------------------------
# 2) MARGIN DRIVER MODEL — Random Forest Regression
# ---------------------------------------------------------------------------
def margin_driver_model(df):
    features = ["Division", "Ship Mode", "Region", "Units", "Sales"]
    target = "Gross_Margin_%"
    data = df[features + [target]].dropna()

    X = data[features]
    y = data[target]

    cat_cols = ["Division", "Ship Mode", "Region"]
    num_cols = ["Units", "Sales"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    pipe = Pipeline([
        ("prep", pre),
        ("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # feature importances (mapped back to readable names)
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    all_names = cat_names + num_cols
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=all_names).sort_values(ascending=False)

    return pipe, mae, r2, fi


# ---------------------------------------------------------------------------
# 3) ANOMALY DETECTION — Isolation Forest
# ---------------------------------------------------------------------------
def detect_anomalies(df, contamination=0.02):
    feats = ["Sales", "Cost", "Units", "Gross_Margin_%"]
    X = df[feats].fillna(0)
    Xs = StandardScaler().fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42)
    df = df.copy()
    df["Anomaly_Flag"] = iso.fit_predict(Xs)   # -1 = anomaly, 1 = normal
    anomalies = df[df["Anomaly_Flag"] == -1]
    return anomalies[["Order ID", "Product Name", "Sales", "Cost", "Units", "Gross_Margin_%"]]


# ---------------------------------------------------------------------------
# 4) SALES FORECASTING — Linear Regression on monthly sales trends
# ---------------------------------------------------------------------------
def forecast_sales(df, periods_ahead=3):
    monthly = (
        df.dropna(subset=["Order Date"])
          .set_index("Order Date")
          .groupby("Division")
          .resample("ME")["Sales"]
          .sum()
          .reset_index()
    )

    forecasts = []
    for div, g in monthly.groupby("Division"):
        g = g.sort_values("Order Date").reset_index(drop=True)
        g["t"] = np.arange(len(g))
        if len(g) < 3:
            continue
        model = LinearRegression().fit(g[["t"]], g["Sales"])
        future_t = np.arange(len(g), len(g) + periods_ahead).reshape(-1, 1)
        preds = model.predict(future_t)
        future_dates = pd.date_range(g["Order Date"].max(), periods=periods_ahead + 1, freq="ME")[1:]
        for d, p in zip(future_dates, preds):
            forecasts.append({"Division": div, "Month": d, "Forecast_Sales": max(p, 0)})

    return monthly, pd.DataFrame(forecasts)


# ---------------------------------------------------------------------------
# 5) DISCONTINUATION-RISK CLASSIFIER — Logistic Regression
# ---------------------------------------------------------------------------
def discontinuation_risk_model(prod):
    sales_med = prod["Total_sales"].median()
    profit_med = prod["Total_profit"].median()
    prod = prod.copy()
    prod["At_Risk"] = ((prod["Total_sales"] < sales_med) & (prod["Total_profit"] < profit_med)).astype(int)

    features = ["Gross_Margin_%", "Cost_to_Sales_%", "Profit_per_Unit", "Revenue_Contribution_%"]
    X = prod[features].fillna(0)
    y = prod["At_Risk"]

    if y.nunique() < 2 or len(prod) < 8:
        return None, "Not enough products/class balance to train a reliable classifier — use the rule-based flag instead."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    report = classification_report(y_test, clf.predict(X_test))
    prod["Risk_Probability"] = clf.predict_proba(X)[:, 1]
    return prod[["Product Name", "At_Risk", "Risk_Probability"]].sort_values("Risk_Probability", ascending=False), report


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    df = load_and_clean()
    prod = build_product_table(df)

    print("=" * 70)
    print("1) PRODUCT SEGMENTATION (KMeans)")
    print("=" * 70)

    prod_seg, scores, best_k = segment_products(prod)

    print(f"Silhouette scores by k: {scores}  -> chosen k = {best_k}")
    print(prod_seg[["Product Name", "Segment", "Gross_Margin_%", "Total_sales"]]
          .sort_values("Segment"))


    print("\n" + "=" * 70)
    print("2) MARGIN DRIVER MODEL (Random Forest Regression)")
    print("=" * 70)

    model, mae, r2, fi = margin_driver_model(df)

    print(f"Test MAE: {mae:.2f} margin points | Test R^2: {r2:.3f}")
    print("Top drivers of Gross Margin %:")
    print(fi.head(10))


    print("\n" + "=" * 70)
    print("3) ANOMALY DETECTION (Isolation Forest)")
    print("=" * 70)

    anomalies = detect_anomalies(df)

    print(f"Flagged {len(anomalies)} anomalous transactions out of {len(df)}")
    print(anomalies.head(15))


    print("\n" + "=" * 70)
    print("4) SALES FORECASTING (Linear trend, 3 months ahead)")
    print("=" * 70)

    monthly, fc = forecast_sales(df)

    print(fc)


    print("\n" + "=" * 70)
    print("5) DISCONTINUATION-RISK CLASSIFIER (Logistic Regression)")
    print("=" * 70)

    risk_table, report = discontinuation_risk_model(prod)

    print(report)

    if risk_table is not None:
        print(risk_table)


    # -----------------------------------------------------------------------
    # SAVE ML OUTPUT FILES
    # -----------------------------------------------------------------------

    prod_seg.to_csv(
        os.path.join(OUTPUT_DIR, "product_segments.csv"),
        index=False
    )

    anomalies.to_csv(
        os.path.join(OUTPUT_DIR, "anomalies.csv"),
        index=False
    )

    fc.to_csv(
        os.path.join(OUTPUT_DIR, "sales_forecast.csv"),
        index=False
    )

    if risk_table is not None:
        risk_table.to_csv(
            os.path.join(OUTPUT_DIR, "discontinuation_risk.csv"),
            index=False
        )

    print("\nML output files saved successfully in:", OUTPUT_DIR)