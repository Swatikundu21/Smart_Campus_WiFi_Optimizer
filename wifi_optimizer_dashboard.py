# ===============================
# 📶 Smart Campus Wi-Fi Optimizer Dashboard
# Streamlit-based professional UI
# Run: streamlit run wifi_optimizer_dashboard.py
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# ---------- App Settings ----------
st.set_page_config(
    page_title="Smart Campus Wi-Fi Optimizer",
    page_icon="📶",
    layout="wide",
)

st.title("📶 Smart Campus Wi-Fi Optimizer")
st.markdown("### Optimize router placement using signal, capacity & budget constraints")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Wi-Fi data (CSV)", type=["csv"])
budget = st.sidebar.number_input("Enter budget (₹)", min_value=100, value=5000, step=100)

# ---------- Utility Functions ----------
def score_location(row):
    return (row['signal_strength'] * 0.5 +
            row['capacity'] * 0.3 -
            row['cost'] * 0.002 +
            row['estimated_users'] * 0.2)

def knapsack(df, budget):
    items = df.to_dict('records')
    n = len(items)
    dp = np.zeros((n + 1, budget + 1))
    
    for i in range(1, n + 1):
        cost = int(items[i - 1]['cost'])
        value = items[i - 1]['score']
        for w in range(budget + 1):
            if cost <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-cost] + value)
            else:
                dp[i][w] = dp[i-1][w]
    
    selected = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(items[i-1])
            w -= int(items[i-1]['cost'])
    return pd.DataFrame(selected)

def minimum_spanning_tree(df):
    coords = df[['x', 'y']].values
    dist_mat = distance_matrix(coords, coords)
    n = len(coords)
    selected = [False] * n
    selected[0] = True
    edges = []
    total_dist = 0

    for _ in range(n - 1):
        min_dist = float('inf')
        x = y = 0
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if not selected[j] and dist_mat[i][j]:
                        if dist_mat[i][j] < min_dist:
                            min_dist = dist_mat[i][j]
                            x, y = i, j
        edges.append((x, y, dist_mat[x][y]))
        total_dist += dist_mat[x][y]
        selected[y] = True
    
    return edges, total_dist

# ---------- Main App Logic ----------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['score'] = df.apply(score_location, axis=1)
    df = df.sort_values('score', ascending=False)

    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

    with st.spinner("Optimizing network placement..."):
        selected_df = knapsack(df, budget)
        edges, total_dist = minimum_spanning_tree(selected_df)

    st.subheader("🏆 Selected Access Points")
    st.dataframe(selected_df, use_container_width=True)

    total_cost = selected_df['cost'].sum()
    st.metric("💰 Total Cost", f"₹{total_cost}")
    st.metric("🧭 Total Cable Distance", f"{total_dist:.2f} units")

    # ---------- Visualization ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df['x'], df['y'], c='lightgray', label='All Locations')
    ax.scatter(selected_df['x'], selected_df['y'], c='blue', label='Selected APs')
    for (x, y, dist) in edges:
        ax.plot([selected_df.iloc[x]['x'], selected_df.iloc[y]['x']],
                [selected_df.iloc[x]['y'], selected_df.iloc[y]['y']], 'r--', linewidth=1)
    ax.legend()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Wi-Fi Access Point Layout')
    st.pyplot(fig)

    # ---------- Download Option ----------
    st.download_button(
        "⬇️ Download Selected APs CSV",
        selected_df.to_csv(index=False).encode('utf-8'),
        "optimized_wifi_locations.csv",
        "text/csv",
        key='download-csv'
    )

else:
    st.info("👆 Upload a CSV file in the sidebar to get started.")
