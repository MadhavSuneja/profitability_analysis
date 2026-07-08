import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Nassau Candy Distributor Analytics", layout="wide")

RAW_PATH = "Nassau Candy Distributor.csv"   # update path as needed

# ---------------------------------------------------------------------------
# Reference data supplied in the requirements doc 
# ---------------------------------------------------------------------------
FACTORY_COORDS = {
    "Lot's O' Nuts":       (32.881893, -111.768036),
    "Wicked Choccy's":     (32.076176,  -81.088371),
    "Sugar Shack":         (48.119140,  -96.181150),
    "Secret Factory":      (41.446333,  -90.565487),
    "The Other Factory":   (35.117500,  -89.971107),
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "Sweetarts":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

# Approximate destination centroids (lat, lon) for US states/provinces present
# in the data — used only to draw indicative factory->destination flow lines
# and estimate relative distance, NOT for precise routing.
STATE_CENTROIDS = {
    "Texas": (31.0, -100.0), "Illinois": (40.0, -89.2), "Pennsylvania": (41.2, -77.6),
    "Kentucky": (37.5, -85.3), "Georgia": (32.6, -83.4), "California": (36.8, -119.6),
    "Virginia": (37.5, -78.6), "Delaware": (39.0, -75.5), "South Carolina": (33.8, -80.9),
    "Ohio": (40.4, -82.9), "Louisiana": (31.0, -92.0), "Oregon": (44.0, -120.5),
    "Arizona": (34.2, -111.6), "Arkansas": (34.9, -92.4), "Michigan": (44.3, -85.6),
    "Tennessee": (35.7, -86.7), "Florida": (27.8, -81.7), "Ontario": (50.0, -85.0),
    "Indiana": (39.8, -86.2), "Nevada": (39.9, -117.2), "New York": (43.0, -75.5),
    "Washington": (47.4, -121.5), "Colorado": (39.0, -105.5), "North Carolina": (35.6, -79.4),
    "New Jersey": (40.1, -74.7), "Massachusetts": (42.3, -71.8), "Missouri": (38.6, -92.6),
    "Alabama": (32.8, -86.8), "Wisconsin": (44.5, -89.5), "Minnesota": (46.0, -94.7),
    "Oklahoma": (35.6, -97.5), "Maryland": (39.0, -76.7), "Utah": (39.3, -111.7),
    "Iowa": (42.0, -93.5), "Mississippi": (32.7, -89.7), "Kansas": (38.5, -98.0),
    "Connecticut": (41.6, -72.7), "Nebraska": (41.5, -99.9), "Idaho": (44.2, -114.5),
    "New Mexico": (34.5, -106.0), "West Virginia": (38.6, -80.6), "Rhode Island": (41.7, -71.5),
    "Montana": (47.0, -110.0), "Maine": (45.3, -69.2), "New Hampshire": (43.9, -71.5),
    "Wyoming": (43.0, -107.5), "Alaska": (64.2, -149.5), "Vermont": (44.0, -72.7),
    "Quebec": (52.0, -72.0), "British Columbia": (53.7, -127.6), "Alberta": (55.0, -115.0),
    "Manitoba": (55.0, -98.0), "Saskatchewan": (52.9, -106.5), "Nova Scotia": (45.0, -63.0),
}


# --------------------------------------------------------------------------- 
# Calculates the shortest distance between two locations on the Earth's surface using the Haversine Formula 
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# STEP 1 — Load & clean (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path):
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
    df = df[(df["Sales"] > 0) & (df["Cost"] >= 0)].dropna(subset=["Sales", "Cost"]).copy()

    df["Profit"] = df["Sales"] - df["Cost"]
    df["Gross_Margin_%"] = df["Profit"] / df["Sales"] * 100
    df["Cost_to_Sales_%"] = df["Cost"] / df["Sales"] * 100
    df["Profit_per_Unit"] = df["Profit"] / df["Units"].replace(0, np.nan)

    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    return df


df_full = load_data(RAW_PATH)

# ---------------------------------------------------------------------------
# SIDEBAR — User Capabilities (date range, division filter, margin slider, search)
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

min_d, max_d = df_full["Order Date"].min(), df_full["Order Date"].max()
date_range = st.sidebar.date_input("Order Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

divisions = st.sidebar.multiselect("Division", sorted(df_full["Division"].unique()),
                                    default=sorted(df_full["Division"].unique()))

margin_threshold = st.sidebar.slider("Minimum Gross Margin %", -50, 100, -50)

search_term = st.sidebar.text_input("Product search")

mask = (
    (df_full["Order Date"] >= pd.to_datetime(date_range[0])) &
    (df_full["Order Date"] <= pd.to_datetime(date_range[1])) &
    (df_full["Division"].isin(divisions)) &
    (df_full["Gross_Margin_%"] >= margin_threshold)
)
if search_term:
    mask &= df_full["Product Name"].str.contains(search_term, case=False, na=False)

df = df_full[mask].copy()

st.title("🍬 Nassau Candy Distributor — Profitability & Logistics Analytics")
st.caption(f"{len(df):,} transactions after filters (of {len(df_full):,} total)")

tabs = st.tabs([
    "Product Profitability", "Division Performance", "Cost vs Margin Diagnostics",
    "Pareto / Concentration", "ML Insights", "Logistics & Factory Correlation"
])

# ---------------------------------------------------------------------------
# TAB 1 — Product Profitability Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Product-Level Margin Leaderboard")
    prod = df.groupby("Product Name").agg(
        Total_sales=("Sales", "sum"), Total_units=("Units", "sum"),
        Total_cost=("Cost", "sum"), Total_profit=("Profit", "sum")
    ).reset_index()
    prod["Gross_Margin_%"] = prod["Total_profit"] / prod["Total_sales"] * 100
    prod["Profit_per_Unit"] = prod["Total_profit"] / prod["Total_units"]
    prod["Revenue_Contribution_%"] = prod["Total_sales"] / prod["Total_sales"].sum() * 100
    prod["Profit_Contribution_%"] = prod["Total_profit"] / prod["Total_profit"].sum() * 100
    prod = prod.sort_values(by="Gross_Margin_%", ascending=False)

    st.dataframe(prod.style.format({
        "Total_sales": "${:,.0f}", "Total_cost": "${:,.0f}", "Total_profit": "${:,.0f}",
        "Gross_Margin_%": "{:.1f}%", "Profit_per_Unit": "${:.2f}",
        "Revenue_Contribution_%": "{:.1f}%", "Profit_Contribution_%": "{:.1f}%"
    }), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(prod, x="Product Name", y="Gross_Margin_%", color="Gross_Margin_%",
                     title="Gross Margin % by Product", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig2 = px.pie(prod, names="Product Name", values="Total_profit",
                      title="Profit Contribution by Product")
        st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# TAB 2 — Division Performance Dashboard
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Division Performance")
    div = df.groupby("Division").agg(
        Total_sales=("Sales", "sum"), Total_cost=("Cost", "sum"), Total_profit=("Profit", "sum")
    ).reset_index()
    div["Gross_Margin_%"] = div["Total_profit"] / div["Total_sales"] * 100
    div["Revenue_Share_%"] = div["Total_sales"] / div["Total_sales"].sum() * 100
    div["Profit_Share_%"] = div["Total_profit"] / div["Total_profit"].sum() * 100
    div["Revenue_Profit_Gap"] = div["Revenue_Share_%"] - div["Profit_Share_%"]

    st.dataframe(div.style.format({
        "Total_sales": "${:,.0f}", "Total_cost": "${:,.0f}", "Total_profit": "${:,.0f}",
        "Gross_Margin_%": "{:.1f}%", "Revenue_Share_%": "{:.1f}%",
        "Profit_Share_%": "{:.1f}%", "Revenue_Profit_Gap": "{:.1f}"
    }), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(data=[
            go.Bar(name="Revenue Share %", x=div["Division"], y=div["Revenue_Share_%"]),
            go.Bar(name="Profit Share %", x=div["Division"], y=div["Profit_Share_%"]),
        ])
        fig.update_layout(barmode="group", title="Revenue vs Profit Share by Division")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig2 = px.box(df, x="Division", y="Gross_Margin_%", title="Margin Distribution by Division")
        st.plotly_chart(fig2, width="stretch")

    margin_med = div["Gross_Margin_%"].median()
    strong = div[(div["Gross_Margin_%"] >= margin_med) & (div["Revenue_Profit_Gap"] <= 0)]
    weak = div[(div["Gross_Margin_%"] < margin_med) & (div["Revenue_Profit_Gap"] > 0)]
    c1, c2 = st.columns(2)
    c1.success(f"Strong efficiency divisions: {', '.join(strong['Division']) or 'None'}")
    c2.warning(f"Structural margin issues: {', '.join(weak['Division']) or 'None'}")

# ---------------------------------------------------------------------------
# TAB 3 — Cost vs Margin Diagnostics
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Cost vs Sales Scatter & Risk Flags")
    pc = df.groupby("Product Name").agg(
        Total_sales=("Sales", "sum"), Total_cost=("Cost", "sum"), Total_units=("Units", "sum")
    ).reset_index()
    pc["Total_profit"] = pc["Total_sales"] - pc["Total_cost"]
    pc["Gross_Margin_%"] = pc["Total_profit"] / pc["Total_sales"] * 100
    pc["Cost_to_Sales_%"] = pc["Total_cost"] / pc["Total_sales"] * 100

    # Scatter plot of Cost vs Sales with reference line at break-even (Cost = Sales)
    fig = px.scatter(pc, x="Total_sales", y="Total_cost", text="Product Name",
                      color="Gross_Margin_%", size="Total_units",
                      title="Cost vs Sales (reference line = break-even)", color_continuous_scale="RdYlGn")
    
    # Add a dashed line representing the break-even point (where Total Cost equals Total Sales)
    maxv = max(pc["Total_sales"].max(), pc["Total_cost"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=maxv, y1=maxv, line=dict(dash="dash"))
    st.plotly_chart(fig, width="stretch")


    # Risk flags for products with high cost-to-sales ratio and low margin
    cost_heavy = pc[(pc["Cost_to_Sales_%"] >= 80) & (pc["Gross_Margin_%"] <= 20)]
    reprice = pc[pc["Gross_Margin_%"] < 15]
    renegotiate = pc[pc["Cost_to_Sales_%"] > 80]

    # Display metrics for flagged products
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost-heavy / margin-poor", len(cost_heavy))
    c2.metric("Needs repricing (<15% margin)", len(reprice))
    c3.metric("Needs cost renegotiation (>80% CTS)", len(renegotiate))
    st.dataframe(pc.sort_values("Gross_Margin_%"), width="stretch")

# ---------------------------------------------------------------------------
# TAB 4 — Pareto / Concentration Analysis
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Revenue & Profit Concentration (Pareto)")
    rev = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).reset_index()
    rev["Cumulative_%"] = rev["Sales"].cumsum() / rev["Sales"].sum() * 100

    fig = go.Figure()
    fig.add_bar(x=rev["Product Name"], y=rev["Sales"], name="Sales")
    fig.add_trace(go.Scatter(x=rev["Product Name"], y=rev["Cumulative_%"], name="Cumulative %",
                              yaxis="y2", mode="lines+markers"))
    fig.update_layout(
        title="Revenue Pareto",
        yaxis2=dict(overlaying="y", side="right", range=[0, 110], title="Cumulative %")
    )
    st.plotly_chart(fig, width="stretch")

    n80 = (rev["Cumulative_%"] >= 80).idxmax() + 1
    st.info(f"{n80} of {len(rev)} products ({n80/len(rev)*100:.1f}%) drive 80% of revenue.")

    st.subheader("State/Province Dependency Risk")
    loc = df.groupby("State/Province").agg(
        Total_sales=("Sales", "sum"), Total_units=("Units", "sum"), Total_profit=("Profit", "sum")
    ).reset_index()
    loc["Revenue_%"] = loc["Total_sales"] / loc["Total_sales"].sum() * 100
    loc["Units_%"] = loc["Total_units"] / loc["Total_units"].sum() * 100
    loc = loc.sort_values("Revenue_%", ascending=False)
    top = loc.iloc[0]
    risk = "High" if top["Revenue_%"] >= 40 else "Moderate" if top["Revenue_%"] >= 25 else "Low"
    st.metric(f"Top location: {top['State/Province']}", f"{top['Revenue_%']:.1f}% of revenue", f"{risk} dependency risk")
    st.dataframe(loc.head(15), width="stretch")

# ---------------------------------------------------------------------------
# TAB 5 — ML Insights
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Product Segmentation (KMeans)")
    prod_ml = df.groupby("Product Name").agg(
        Total_sales=("Sales", "sum"), Total_profit=("Profit", "sum"), Total_units=("Units", "sum")
    ).reset_index()
    prod_ml["Gross_Margin_%"] = prod_ml["Total_profit"] / prod_ml["Total_sales"] * 100
    prod_ml["Profit_per_Unit"] = prod_ml["Total_profit"] / prod_ml["Total_units"]
    prod_ml["Revenue_Contribution_%"] = prod_ml["Total_sales"] / prod_ml["Total_sales"].sum() * 100


    # Feature selection for clustering and scaling
    feats = ["Gross_Margin_%", "Total_sales", "Profit_per_Unit", "Revenue_Contribution_%"]
    
    # Only perform clustering if there are enough distinct products
    if len(prod_ml) >= 4:
        Xscaled = StandardScaler().fit_transform(prod_ml[feats].fillna(0))
        k = st.slider("Number of segments (k)", 2, min(6, len(prod_ml) - 1), 4)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xscaled)
        prod_ml["Segment"] = labels
        fig = px.scatter(prod_ml, x="Total_sales", y="Gross_Margin_%", color=prod_ml["Segment"].astype(str),
                          size="Total_units", text="Product Name", title="Product Segments")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(prod_ml.sort_values("Segment"), width="stretch")
    else:
        st.warning("Not enough distinct products in the current filter to cluster.")

    st.subheader("Anomaly Detection (Isolation Forest)")
    contamination = st.slider("Expected anomaly rate", 0.01, 0.10, 0.02)
    afeats = ["Sales", "Cost", "Units", "Gross_Margin_%"]
    Xscaled_a = StandardScaler().fit_transform(df[afeats].fillna(0))
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["Anomaly"] = iso.fit_predict(Xscaled_a)
    anomalies = df[df["Anomaly"] == -1]
    st.write(f"Flagged {len(anomalies)} anomalous transactions ({len(anomalies)/len(df)*100:.1f}%)")
    st.dataframe(anomalies[["Order ID", "Product Name", "Sales", "Cost", "Units", "Gross_Margin_%"]],
                 width="stretch")

    st.subheader("Sales Forecast (Linear Trend, 3 months)")
    monthly = df.dropna(subset=["Order Date"]).set_index("Order Date").groupby("Division").resample("ME")["Sales"].sum().reset_index()
    fig = px.line(monthly, x="Order Date", y="Sales", color="Division", title="Monthly Sales Trend by Division")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# TAB 6 — Logistics & Factory Correlation
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Factory Network & Product Sourcing")
    st.caption("Uses the Factory coordinates and Product→Factory mapping supplied in the project requirements.")

    fac_summary = df.groupby("Factory").agg(
        Total_sales=("Sales", "sum"), Total_units=("Units", "sum"), Total_profit=("Profit", "sum")
    ).reset_index()
    fac_summary["Revenue_Share_%"] = fac_summary["Total_sales"] / fac_summary["Total_sales"].sum() * 100
    fac_summary["lat"] = fac_summary["Factory"].map(lambda f: FACTORY_COORDS.get(f, (None, None))[0])
    fac_summary["lon"] = fac_summary["Factory"].map(lambda f: FACTORY_COORDS.get(f, (None, None))[1])

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter_geo(fac_summary, lat="lat", lon="lon", size="Total_sales",
                              color="Revenue_Share_%", hover_name="Factory",
                              scope="north america", title="Factory Output Volume (by Sales)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(fac_summary.style.format({
            "Total_sales": "${:,.0f}", "Total_profit": "${:,.0f}", "Revenue_Share_%": "{:.1f}%"
        }), width="stretch")

    top_fac = fac_summary.sort_values("Revenue_Share_%", ascending=False).iloc[0]
    risk = "High" if top_fac["Revenue_Share_%"] >= 40 else "Moderate" if top_fac["Revenue_Share_%"] >= 25 else "Low"
    # Display a metric for the most relied-upon factory and its associated risk
    st.metric(f"Most relied-upon factory: {top_fac['Factory']}", f"{top_fac['Revenue_Share_%']:.1f}% of revenue",
              f"{risk} single-source dependency risk")

    st.subheader("Indicative Factory → Destination Flow")
    flow = df.dropna(subset=["Factory"]).groupby(["Factory", "State/Province"]).agg(
        Total_sales=("Sales", "sum")
    ).reset_index()
    flow["f_lat"] = flow["Factory"].map(lambda f: FACTORY_COORDS.get(f, (None, None))[0])
    flow["f_lon"] = flow["Factory"].map(lambda f: FACTORY_COORDS.get(f, (None, None))[1])
    flow["s_lat"] = flow["State/Province"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[0])
    flow["s_lon"] = flow["State/Province"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[1])
    flow = flow.dropna(subset=["f_lat", "s_lat"])
    flow["Distance_km"] = haversine(flow["f_lat"], flow["f_lon"], flow["s_lat"], flow["s_lon"])

    top_flows = flow.sort_values("Total_sales", ascending=False).head(25)
    fig = go.Figure()
    for _, r in top_flows.iterrows():
        fig.add_trace(go.Scattergeo(
            lat=[r["f_lat"], r["s_lat"]], lon=[r["f_lon"], r["s_lon"]],
            mode="lines", line=dict(width=1, color="crimson"), opacity=0.4, showlegend=False
        ))
    fig.update_geos(scope="north america")
    fig.update_layout(title="Top 25 Factory → State Sales Flows (line width not to scale)")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Note: destination points are approximate state/province centroids, not exact delivery "
        "addresses, so distances are indicative only — useful for spotting which factories serve "
        "distant regions, not for precise route optimization."
    )

    corr = flow.groupby("Factory").agg(Avg_Distance_km=("Distance_km", "mean")).reset_index().merge(
        fac_summary[["Factory", "Total_sales", "Revenue_Share_%"]], on="Factory"
    )
    st.dataframe(corr.style.format({"Avg_Distance_km": "{:,.0f}", "Total_sales": "${:,.0f}", "Revenue_Share_%": "{:.1f}%"}),
                 width="stretch")

    st.warning(
        "Data quality note: the raw file's Ship Date values are 3-4 years after the matching Order "
        "Date on every row, which is not a plausible real shipment lag. Treat this as a placeholder "
        "field rather than a real fulfilment-time metric until it is corrected at the source."
    )