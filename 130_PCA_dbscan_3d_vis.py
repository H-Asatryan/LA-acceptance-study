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

# # 3D interactive visualization for DBSCAN Clustering with PCA

# ## Import packages and data

# +
import pandas as pd
import numpy as np

# # %matplotlib notebook
# %matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider #, Button, RadioButtons
# import matplotlib
# matplotlib.use('TkAgg')
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler,MinMaxScaler

# +
# Load imputed data
acceptance_imputed_df = pd.read_excel(
    './data/clean_data/data_imputed.xlsx',
    index_col="SurveyID"
)

# Drop the superfluous columns
acceptance_imputed_df = acceptance_imputed_df.drop(
    acceptance_imputed_df.iloc[:, 1:4], axis=1)

# Exclude the categorical (integer) column of specializations
acceptance_num = acceptance_imputed_df.iloc[:, 1:]
# -

# ## PCA

# ### Scaling

# Scaling numerical features for PCA
scaler = RobustScaler()
acceptance_scaled = pd.DataFrame(scaler.fit_transform(acceptance_num),
                                 columns=acceptance_num.columns)

# ### PCA

# +
n_pcs = 3 # Threshold PCA
pca = PCA(n_components=n_pcs, whiten=True)
pca.fit(acceptance_scaled)
acceptance_proj = pd.DataFrame(pca.transform(acceptance_scaled))
# acceptance_proj.head(2)

# This command also prevents NaN values after column assignment
acceptance_proj = acceptance_proj.set_index(acceptance_imputed_df.index)
# Naming PCA features for plotly visualizations
acceptance_proj.columns=["pr_PC1","pr_PC2","pr_PC3"]


# -

# ## DBSCAN clustering with 3 PCA columns

def get_dbscan_cluster_colors(epsilon):
    dbscan_df = acceptance_proj.copy()
    clustering = DBSCAN(eps=epsilon).fit(acceptance_proj)
    dbscan_df["DBSCAN_clusters"] = clustering.labels_
    dbscan_df['DBSCAN_clusters'] = dbscan_df['DBSCAN_clusters'].astype("str")
    # dbscan_df.head(2)

    cluster_nums = dbscan_df['DBSCAN_clusters'].value_counts(
    ).rename_axis('DBSCAN_clusters').reset_index(name='count')

    transformed_cluster_no = MinMaxScaler(
    ).fit_transform(cluster_nums[["DBSCAN_clusters"]]) # to get in range [0,1]

    cm = plt.get_cmap("RdYlGn")
    cluster_color_list = [cm(i) for i in transformed_cluster_no]

    color_dic = dict(
        zip(cluster_nums["DBSCAN_clusters"],cluster_color_list)
    )

    plt_colors_vec = [color_dic[cluster_idx] for cluster_idx in dbscan_df['DBSCAN_clusters']]
    return plt_colors_vec


# +
eps_start = 0.43
eps_end = 0.475
eps_step = 0.05
plt.close("all")
fig = plt.figure()
# fig.tight_layout()
ax = fig.add_subplot(projection='3d')
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
ax.scatter(acceptance_proj["pr_PC1"],
           acceptance_proj["pr_PC2"],
           acceptance_proj["pr_PC3"],
           c=get_dbscan_cluster_colors(eps_start),
           marker='o',
           s=20)
fig.subplots_adjust(left=0.1, top=0.959, right=0.99, bottom = 0.15)

# Create axes for the epsilon slider
ax_eps = plt.axes([0.25, 0.05, 0.6, 0.03])

# Create a slider from eps_start to eps_step in axes ax_eps
# fig.add_subplot()
eps_slider = Slider(ax_eps, 'DBSCAN Epsilon  ', eps_start, eps_end, eps_step)

# Create function to be called when slider value is changed
def update(val):
    eps_value = eps_slider.val
    ax.cla()
    ax.scatter(acceptance_proj["pr_PC1"],
               acceptance_proj["pr_PC2"],
               acceptance_proj["pr_PC3"],
               c=get_dbscan_cluster_colors(eps_value),
               marker='o',
               s=20)
    fig.canvas.draw_idle()

# Call update function when slider value is changed
eps_slider.on_changed(update)

# display graph
plt.title("DBSCAN Clustering with 3 PCAs",y=29.5)
# plt.savefig("./output/clustering/dbscan/clusters_dbscan_3d_v2.pdf")
plt.show()
