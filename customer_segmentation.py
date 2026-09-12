import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "sample_-_superstore.xls"

print("Loading dataset...")

orders = pd.read_excel(
    file_path,
    sheet_name="Orders",
    engine="xlrd"
)

print("Dataset loaded successfully!")
print("Number of rows:", len(orders))
print("Number of columns:", len(orders.columns))


# ============================================================
# 2. DATA PREPARATION
# ============================================================

orders["Order Date"] = pd.to_datetime(orders["Order Date"])

# Reference date = one day after the latest order
reference_date = orders["Order Date"].max() + pd.Timedelta(days=1)


# ============================================================
# 3. CREATE RFM FEATURES
# ============================================================

print("\nCreating RFM features...")

rfm = orders.groupby("Customer ID").agg(
    Recency=("Order Date", lambda x: (reference_date - x.max()).days),
    Frequency=("Order ID", "nunique"),
    Monetary=("Sales", "sum")
).reset_index()

print("Number of customers:", len(rfm))


# ============================================================
# 4. HANDLE SKEWNESS
# ============================================================

rfm_features = ["Recency", "Frequency", "Monetary"]

rfm_log = rfm[rfm_features].copy()

rfm_log = np.log1p(rfm_log)


# ============================================================
# 5. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_log)


# ============================================================
# 6. FIND BEST NUMBER OF CLUSTERS
# ============================================================

print("\nTesting different numbers of clusters...")

inertia = []
silhouette_scores = []

k_values = range(2, 7)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(rfm_scaled)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(rfm_scaled, labels)
    )

    print(
        f"K = {k} | "
        f"Inertia = {kmeans.inertia_:.2f} | "
        f"Silhouette Score = {silhouette_scores[-1]:.3f}"
    )


# Select the cluster count with the highest silhouette score
best_k = list(k_values)[np.argmax(silhouette_scores)]

print("\nBest number of clusters:", best_k)


# ============================================================
# 7. ELBOW METHOD GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "elbow_method.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. SILHOUETTE SCORE GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "silhouette_scores.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. FINAL K-MEANS MODEL
# ============================================================

print("\nTraining final K-Means model...")

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = final_model.fit_predict(rfm_scaled)


# ============================================================
# 10. CUSTOMER SEGMENT NAMES
# ============================================================

cluster_profile = rfm.groupby("Cluster")[rfm_features].mean()

# Rank clusters based on Recency, Frequency and Monetary values

cluster_profile["Score"] = (
    cluster_profile["Frequency"].rank(pct=True)
    + cluster_profile["Monetary"].rank(pct=True)
    + (1 - cluster_profile["Recency"].rank(pct=True))
)

cluster_order = cluster_profile.sort_values(
    "Score",
    ascending=False
).index.tolist()


segment_names = {}

if len(cluster_order) >= 1:
    segment_names[cluster_order[0]] = "Champions"

if len(cluster_order) >= 2:
    segment_names[cluster_order[1]] = "Loyal Customers"

if len(cluster_order) >= 3:
    segment_names[cluster_order[2]] = "Potential Customers"

if len(cluster_order) >= 4:
    segment_names[cluster_order[3]] = "At Risk Customers"

if len(cluster_order) >= 5:
    segment_names[cluster_order[4]] = "Low Value Customers"

if len(cluster_order) >= 6:
    segment_names[cluster_order[5]] = "Inactive Customers"


rfm["Segment"] = rfm["Cluster"].map(segment_names)


# ============================================================
# 11. ADD AVAILABLE CUSTOMER ATTRIBUTES
# ============================================================

customer_attributes = orders.groupby("Customer ID").agg(
    Region=("Region", lambda x: x.mode()[0]),
    Customer_Type=("Segment", lambda x: x.mode()[0])
).reset_index()

rfm = rfm.merge(
    customer_attributes,
    on="Customer ID",
    how="left"
)


# ============================================================
# 12. SAVE CUSTOMER SEGMENTS
# ============================================================

rfm.to_csv(
    "customer_segments.csv",
    index=False
)

print("\nCustomer segmentation file saved:")
print("customer_segments.csv")


# ============================================================
# 13. SEGMENT SUMMARY
# ============================================================

segment_summary = rfm.groupby("Segment").agg(
    Customers=("Customer ID", "count"),
    Average_Recency=("Recency", "mean"),
    Average_Frequency=("Frequency", "mean"),
    Average_Monetary=("Monetary", "mean")
).reset_index()

segment_summary = segment_summary.sort_values(
    "Average_Monetary",
    ascending=False
)

segment_summary.to_csv(
    "segment_summary.csv",
    index=False
)


# ============================================================
# 14. DISPLAY SEGMENT SUMMARY
# ============================================================

print("\n================ SEGMENT SUMMARY ================\n")

print(
    segment_summary.to_string(
        index=False
    )
)


# ============================================================
# 15. CUSTOMER SEGMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

sns.countplot(
    data=rfm,
    x="Segment",
    order=rfm["Segment"].value_counts().index
)

plt.title("Customer Segment Distribution")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "segment_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. FREQUENCY VS MONETARY VALUE
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Segment",
    s=80
)

plt.title("Customer Segmentation: Frequency vs Monetary Value")
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Sales / Monetary Value")

plt.tight_layout()

plt.savefig(
    "customer_segments_scatter.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. RFM HEATMAP
# ============================================================

heatmap_data = segment_summary.set_index(
    "Segment"
)[
    [
        "Average_Recency",
        "Average_Frequency",
        "Average_Monetary"
    ]
]

# Normalize each column for easier comparison
heatmap_normalized = (
    heatmap_data - heatmap_data.min()
) / (
    heatmap_data.max() - heatmap_data.min()
)

plt.figure(figsize=(9, 6))

sns.heatmap(
    heatmap_normalized,
    annot=True,
    fmt=".2f"
)

plt.title("RFM Segment Profile Heatmap")
plt.xlabel("RFM Metrics")
plt.ylabel("Customer Segment")

plt.tight_layout()

plt.savefig(
    "segment_profile_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n================================================")
print("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
print("================================================")

print("\nFiles created:")

print("1. customer_segments.csv")
print("2. segment_summary.csv")
print("3. elbow_method.png")
print("4. silhouette_scores.png")
print("5. segment_distribution.png")
print("6. customer_segments_scatter.png")
print("7. segment_profile_heatmap.png")

print("\nProject uses:")
print("- RFM Analysis")
print("- K-Means Clustering")
print("- StandardScaler")
print("- Silhouette Score")
print("- Customer Segment Analysis")
print("- Data Visualization")