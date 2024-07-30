# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Study acceptance: PCA and DBSCAN/Spectral Clustering

# ## Import packages and data

# +
import pandas as pd
import numpy as np
from math import ceil

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, SpectralClustering
import hdbscan

# pip install -U kaleido # to export plotly figures as static images

# +
# Load imputed data
acceptance_imputed_df = pd.read_excel('./data/clean_data/data_imputed.xlsx',index_col="SurveyID")

row_mean_features_df = pd.read_csv("./data/clean_data/row_mean_features.csv",index_col="SurveyID")
# Drop the general LA columns and keep the partitioned LA ones
# (otherwise we have strong correlations etc)
row_mean_features_df = row_mean_features_df.drop(columns=["LA Desire", "LA Expectation"])

# Drop the superfluous columns
acceptance_imputed_df = acceptance_imputed_df.drop(
    acceptance_imputed_df.iloc[:, 1:4], axis=1)

# Exclude the categorical (integer) column,
# corresponding to specializations
acceptance_num = acceptance_imputed_df.iloc[:, 1:]
# -

# ## PCA

# ### Scaling

# Scaling numerical features for PCA
scaler = RobustScaler()
acceptance_scaled = pd.DataFrame(scaler.fit_transform(acceptance_num),
                                 columns=acceptance_num.columns)

# ### Cumulated explained variance

pca = PCA()
pca.fit(acceptance_scaled); # find all the Principal Components PC (no y!)

# The Elbow Method for PCA
with plt.style.context('seaborn-deep'):
    # figsize
    plt.figure(figsize=(9,5))
    # getting axes
    ax = plt.gca()
    # plotting
    explained_variance_ratio_cumulated = np.cumsum(pca.explained_variance_ratio_)
    x_axis_ticks = np.arange(1,explained_variance_ratio_cumulated.shape[0]+1)
    ax.plot(x_axis_ticks,explained_variance_ratio_cumulated,
            label="cumulated variance ratio",color="purple",linestyle=":",marker="D",markersize=10)
    # customizing
    ax.set_xlabel('Number of Principal Components',size=11,labelpad=14)
    ax.set_ylabel('Cumulated explained variance (ratio)',size=11,labelpad=10)
    ax.legend(loc="upper left")
    ax.set_title('The Elbow Method')
    ax.set_xticks(x_axis_ticks)
    ax.scatter(5,explained_variance_ratio_cumulated[5-1],c='blue',s=400)
    ax.grid(axis="x",linewidth=0.5)
    ax.grid(axis="y",linewidth=0.5)

# We see an elbow at `n_pcs = 5`, but to deal with DBSCAN later, we need to choose $n_{PCs} \leq 3$. <font color="red">For our small data set, we take $n_{PCs} = 2$. </font> Then we'll project our `acceptance_scaled` dataset onto the new space with the number of principal components we decided to keep. We name it `acceptance_proj`.

# +
n_pcs = 2 # Threshold PCA, choose either 2 or 3
pca = PCA(n_components=n_pcs, whiten=True)
pca.fit(acceptance_scaled)
acceptance_proj = pd.DataFrame(pca.transform(acceptance_scaled))
# acceptance_proj.head(2)

# This command also prevents NaN values after column assignment
acceptance_proj = acceptance_proj.set_index(acceptance_imputed_df.index)
# Naming PCA features for plotly visualizations
acceptance_proj.columns = ["pr_PC"+str(i+1) for i in range(n_pcs)]
# -

# ## DBSCAN clustering with 2-3 PCA columns

# ### Choosing DBSCAN Parameters

# Compute data proximity from each other using Nearest Neighbours
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(acceptance_proj)
distances, indices = nbrs.kneighbors(acceptance_proj)
plt.hist(distances.max(axis=1), bins=20, edgecolor='w')
plt.xlabel('Max distances to 2 neighbors');

plt.plot(np.sort(distances, axis=0)[:,1]);

# For $n_{pcs} = 2$, the maximum curvature of the curve corresponds to the ordinate (about) 0.25, which suggests $\varepsilon\approx0.25$ for DBSCAN. To find a better match, we perform a search:

# Searching epsilons in the region of max curvature
epsils = np.linspace(0.2425,0.2463,num=10)
# epsils = [0.2282, 0.2287, 0.2303]
iterations = len(epsils)
dbscan_df = acceptance_imputed_df.copy()
for i in range(iterations):
    clustering = DBSCAN(eps=epsils[i]).fit(acceptance_proj)
    dbscan_df["DBSCAN_clusters"] = clustering.labels_
    print("eps =", epsils[i], "| The number of clusters (incl '-1'):",
      len(dbscan_df["DBSCAN_clusters"].value_counts())
     )

# +
eps_range = np.arange(0.2425,0.2463, 0.0001)
clusters = []
noise = []
for eps in eps_range:
    dbscan = DBSCAN(eps=eps)
    labels = dbscan.fit_predict(acceptance_proj)
    clusters.append(np.sum(np.unique(labels) >= 0))
    noise.append(100 * np.sum(labels == -1) / len(labels))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(eps_range, clusters)
ax1.set_xlabel(r'DBSCAN $\varepsilon$')
ax1.set_ylabel('#clusters')
ax2.plot(eps_range, noise, c='gray')
ax2.set_ylabel('Noise in %');

# +
last_three = clusters[::-1].index(3)
best_eps = eps_range[-last_three - 1]
dbscan = DBSCAN(eps=best_eps)
labels = dbscan.fit_predict(acceptance_proj)

fig, ax = plt.subplots()
for c_id in np.unique(labels):
    if c_id < 0:
        # skip noise
        continue

    cluster_idx = labels == c_id
    questions_cluster = acceptance_num[cluster_idx]
    ax.plot(range(acceptance_num.shape[1]), questions_cluster.mean())
# fig.savefig('./output/clustering/dbscan/clusters_lineplot'+"d.jpg",
#             bbox_inches='tight', pad_inches=0,format="jpg")
# scale > 1 improves the resolution
# -

# We may take $\varepsilon = 0.2451$ and try to view cluster sizes:

clustering = DBSCAN(eps=0.2451).fit(acceptance_proj)
dbscan_df["DBSCAN_clusters"] = clustering.labels_
dbscan_df["DBSCAN_clusters"].value_counts()

# We see a dominating cluster (442 elements), the distribution does not seem to be satisfactory! So we'll try to get rid of the cluster "-1" (unclassified elements).

# Looking for unclassified objects (group labeled "-1")
epsils = np.linspace(1.008,1.009,num=10)
# epsils = [0.2282, 0.2287, 0.2303]
iterations = len(epsils)
for i in range(iterations):
    clustering = DBSCAN(eps=epsils[i]).fit(acceptance_proj)
    dbscan_df["DBSCAN_clusters"] = clustering.labels_
    dbscan_df['DBSCAN_clusters'] = dbscan_df['DBSCAN_clusters'].astype("str")
    dbsc_label_counts = dbscan_df["DBSCAN_clusters"].value_counts()
    if "-1" in dbsc_label_counts.keys():
        print(
        "eps =", epsils[i], "| Unclassified objects:", dbsc_label_counts["-1"]
         )

# We choose $\varepsilon = 1.009$ to have only classified elements:

clustering = DBSCAN(eps=1.009).fit(acceptance_proj)
dbscan_df["DBSCAN_clusters"] = clustering.labels_
dbscan_df["DBSCAN_clusters"].value_counts()

# Now we have only one cluster, much worse! Let's give another try, focusing on constructing of 4 groups:

epsils = np.linspace(0.24465,0.2455,num=10)
for eps in epsils:
    clustering = DBSCAN(eps=eps).fit(acceptance_proj)
    dbscan_df["DBSCAN_clusters"] = clustering.labels_
    print("DBSCAN clusters for eps=",eps,":\n", dbscan_df["DBSCAN_clusters"].value_counts(),"\n",sep="")

# We see that the cluster sizes essentially do not change. Again, we observe a bad distribution.

clustering = DBSCAN(eps=0.245).fit(acceptance_proj) # eps=0.245 for 2d and eps=0.47 for 3d
dbscan_df["DBSCAN_clusters"] = clustering.labels_
dbscan_df["DBSCAN_clusters"].value_counts()

# Sorting clusters by size
# Get value counts as a 2d array
label_counts = np.asarray(np.unique(clustering.labels_, return_counts=True)).T
# Sort label_counts in descending order by the second column
label_counts_sorted_desc = label_counts[label_counts[:, 1].argsort()[::-1]]
# Dictionary to set new cluster labels
label_convert_dict = dict(
    zip(label_counts_sorted_desc[:, 0],
        np.sort(label_counts_sorted_desc[:,0])))
# Translate labels according to the dictionary label_convert_dict
sorted_labels_dbscan_pca = np.vectorize(
    label_convert_dict.get)(clustering.labels_)
dbscan_df["DBSCAN_clusters"] = sorted_labels_dbscan_pca
dbscan_df["DBSCAN_clusters"].value_counts() # check-up new label counts

# ### Plot the clusters:

dbscan_plot_df = acceptance_proj.copy()
dbscan_plot_df["DBSCAN_clusters"] = dbscan_df["DBSCAN_clusters"]
dbscan_plot_df['DBSCAN_clusters'] = dbscan_plot_df['DBSCAN_clusters'].astype("str")
dbscan_plot_df['DBSCAN_clusters'].value_counts()

# +
# Plotly visualization: A cluster plot that can be rotated etc
if n_pcs==2:
    dbscan_fig = px.scatter(dbscan_plot_df,
                            x=dbscan_plot_df["pr_PC1"],
                            y=dbscan_plot_df["pr_PC2"],
                            color='DBSCAN_clusters',
                            width=500,
                            height=420,
#                             template='plotly_dark',
                            title='2d Scatter Plot for DBSCAN Clusters')

if n_pcs==3:
    dbscan_fig = px.scatter_3d(dbscan_plot_df,
                               x=dbscan_plot_df["pr_PC1"],
                               y=dbscan_plot_df["pr_PC2"],
                               z=dbscan_plot_df["pr_PC3"],
                               color='DBSCAN_clusters',
                               width=500,
                               height=500,
#                                template='plotly_dark',
                               title='3D Scatter plot for DBSCAN Clusters')

dbscan_fig.update_traces(marker_size=6)  # reduce marker sizes
dbscan_fig.update_layout(hovermode=False)  # remove hover info for x, y and z (in 3d mode)
dbscan_fig.show()

dbscan_fig.write_html("./output/clustering/dbscan/clusters_dbscan_"+str(n_pcs)+"d.html")  # interactive plot
dbscan_fig.write_image(file="./output/clustering/dbscan/clusters_dbscan_"+str(n_pcs)+"d.jpg",
                       format="jpg",
                       scale=3)  # scale > 1 improves the resolution
# -

# ### Grade averages for the clusters

# +
unique_clusters = np.unique(clustering.labels_)
n_clusters = len(dbscan_df["DBSCAN_clusters"].value_counts())
dbscan_df_cluster_list = [dbscan_df[
    dbscan_df["DBSCAN_clusters"]==i].iloc[:,:-1] for i in unique_clusters]

# Compute cluster sizes
dbscan_cluster_sizes = np.array([len(df) for df in dbscan_df_cluster_list])

# +
xlab_angle = 55
dbscan_df.iloc[:, 1:-1].mean().plot(figsize=(10, 4),
                                    color="black",
                                    xticks=range(dbscan_df.iloc[:, 1:-1].shape[1]),
                                    rot=xlab_angle,
                                    title="Averages per Group (DBSCAN with "+str(n_pcs)+" PCs)",
                                    label="Total ("+str(len(dbscan_df))+")")

for idx, df in enumerate(dbscan_df_cluster_list):
    df.iloc[:, 1:].mean().plot(
        label="Group " + str(idx + 1)+" ("+str(len(df))+")",
        rot=xlab_angle
    )

plt.legend(loc="lower left", bbox_to_anchor=(1, 0.6))
plt.grid(linestyle='dotted', linewidth=0.4)
plt.gca().set_xlabel('Question code', size = 12, labelpad=15)
plt.gca().set_ylabel('Average grade', size = 12, labelpad=15)
plt.savefig("./output/clustering/dbscan/"+"averages_groups_dbscan_"+str(n_pcs)+"d" + ".jpg",
            format="jpg",
            bbox_inches="tight",
            dpi=200)
# -

# We do not get well interpretable groups!

# ### Average Scores per Group and Question

# +
# Prepare the visualization data set based on row means features per cluster

dbscan_merged_row_means = dbscan_df.iloc[:, -1:].join(row_mean_features_df).dropna()
# dbscan_merged_row_means.head()

# Calculate feature averages in total / within each cluster
# and save them to a list
mean_feat_total_avg = dbscan_merged_row_means.iloc[:, 1:].mean()
means_list = [mean_feat_total_avg]

for idx in unique_clusters:
    means_list.append(dbscan_merged_row_means.loc[dbscan_merged_row_means["DBSCAN_clusters"] == idx].iloc[:,1:].mean())
means_group_df = pd.concat(means_list, axis=1).T

# Create a detailed list index, add group IDs and sizes to the index
# Uncomment one of the following lines to display clusters by their number or description,
# respectively (to reveal the descriptions we refer to the discussion after plots)
cluster_names = np.char.add(np.repeat("Group ",n_clusters),np.arange(1,5).astype(str))
# cluster_names = ["Enthusiasts","Realists","Indifferent", "Cautious"]

means_groups_idx_vec = cluster_names
means_groups_idx_vec = np.char.add(means_groups_idx_vec,np.repeat(" (",n_clusters))
means_groups_idx_vec = np.char.add(means_groups_idx_vec,dbscan_cluster_sizes.astype(str))
means_groups_idx_vec = np.char.add(means_groups_idx_vec,np.repeat(")",n_clusters))
means_groups_index = ["All together ("+str(len(dbscan_df))+")"]+list(means_groups_idx_vec)
means_group_df=means_group_df.set_index(pd.Index(means_groups_index))
means_group_df

# +
# Bar chart for feature averages in total / within each cluster
fig_desire_expec_avg_group = means_group_df.iloc[:,0:6].plot.bar(
    figsize=(8, 2.5),
    width=0.65,
    rot=0,
    linewidth=2,
    color=['#4F81BD', '#C0504D',"cadetblue","LightCoral","#00CDCD","#8B008B"],
    edgecolor='white'
)

# Customizing
fig_desire_expec_avg_group.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_desire_expec_avg_group.set_xlabel('Student groups', size="medium", labelpad=14)
fig_desire_expec_avg_group.set_ylabel('Average grades', size = "medium", labelpad=20)
fig_desire_expec_avg_group.tick_params(axis='x', labelsize="small")
fig_desire_expec_avg_group.set_title("Grade Averages per Group (DBSCAN with "+str(n_pcs)+" PCs)", size=12, pad=12)

# Alter legend's placement
fig_desire_expec_avg_group.legend(
    loc="center left",
    fontsize = "x-small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (1, 0.7)
)

fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/dbscan/"+"desire_exp_avg_dbscan_"+str(n_pcs)+"_PCs.jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/dbscan/"+"desire_exp_avg_dbscan_"+str(n_pcs)+"_PCs.pdf",
    format="pdf", bbox_inches="tight"
);
# -

# ## HDBSCAN clustering with 2 PCA columns

# +
alg = hdbscan.HDBSCAN(
    min_samples=3,
    min_cluster_size=6,
    cluster_selection_epsilon=0.245,
    allow_single_cluster=True,
    prediction_data=True)

alg.fit(acceptance_proj)
# alg.labels_
# -

# Plotly visualization: A cluster plot that can be rotated etc
# As x,y,z we select uncorrelated variables
fig = px.scatter(acceptance_proj,
                 x=acceptance_proj["pr_PC1"],
                 y=acceptance_proj["pr_PC2"],
                 color=alg.labels_,
                 width=500,
                 height=500,
                 title='2D Scatter plot for HDBSCAN Clusters')
fig.update_traces(marker_size=6)  # reduce marker sizes
fig.update_layout(hovermode=False)  # remove hover info for x, y, z
fig.show()
fig.write_html("./output/clustering/dbscan/clusters_hdbscan.html")  # interactive plot
fig.write_image(file="./output/clustering/dbscan/clusters_hdbscan.jpg",
                format="jpg",
                scale=3)  # scale > 1 improves the resolution

# Like DBSCAN, we observe a bad distribution.

# ## Spectral clustering with 3 PCA columns

# +
n_pcs = 3 # Threshold PCA, choose either 2 or 3
pca = PCA(n_components=n_pcs, whiten=True)
pca.fit(acceptance_scaled)
acceptance_proj = pd.DataFrame(pca.transform(acceptance_scaled))
# acceptance_proj.head(2)

# This command also prevents NaN values after column assignment
acceptance_proj = acceptance_proj.set_index(acceptance_imputed_df.index)
# Naming PCA features for plotly visualizations
acceptance_proj.columns = ["pr_PC"+str(i+1) for i in range(n_pcs)]
# -

spec_clustering = SpectralClustering(n_clusters=3,
                                     random_state=0,
                                     gamma = 1) # >=1, 1 is the default and the best, adjust here
spec_clustering.fit(acceptance_proj)
spec_label_counts = np.asarray(np.unique(spec_clustering.labels_, return_counts=True))[1,:].tolist()
print("Number of clusters:",len(spec_label_counts),"\nCluster sizes:", *spec_label_counts)

# Plotly visualization: A cluster plot that can be rotated etc
# As x,y,z we select uncorrelated variables
fig = px.scatter_3d(acceptance_proj,
                    x=acceptance_proj["pr_PC1"],
                    y=acceptance_proj["pr_PC2"],
                    z=acceptance_proj["pr_PC3"],
                    color=spec_clustering.labels_,
                    width=500,
                    height=500,
                    title='3D Scatter plot for Spectral Clusters')
fig.update_traces(marker_size=3)  # reduce marker sizes
fig.update_layout(hovermode=False)  # remove hover info for x, y, z
fig.show()
fig.write_html("./output/clustering/spectral/clusters_spectral.html")  # interactive plot
fig.write_image(file="./output/clustering/spectral/clusters_spectral.jpg", format="jpg",
                scale=3)  # scale > 1 improves the resolution

# Again, we observe a bad distribution.
