# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Auxiliary functions and variables

# %% [markdown]
# <span style="color: red">Caution:</span> Do not use a filename beginning with a digit to avoid importing problems!

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import os

# %%
specializations =[
 'Architektur','Bauingenieurwesen','Elektrotechnik','Informatik','Maschinenbau','Mechatronik',
    'Nachhaltigkeit','Vermessungswesen','Wirtschaftswissenschaften','Sonstige Ingenieurstudiengänge'
]

specializations_en =[
 'Architecture', 'Civil engineering', 'Electrical engineering', 'Informatics', 'Mechanical engineering',
    'Mechatronics', 'Sustainability', 'Surveying', 'Business studies', 'Other engineering courses'
]

# Merge "Electrical engineering", "Mechanical engineering" and "Mechatronics" (small groups)
specializations_merged =[
 'Architecture', 'Civil Engineering', 'Electro-Mechanical Engineering', 'Computer Science',
    'Sustainability', 'Surveying', 'Business Studies', 'Other Engineering Courses'
]


# %%
# Create the directory on demand
def check_and_create_directory(directory_name):
  """Checks if the directory exists and creates it if it doesn't."""
  if not os.path.exists(directory_name):
    os.mkdir(directory_name)


# %%
# A function to plot seaborn histograms for all data frame variables
def plot_df_histograms(dataframe, output_name="histograms_all", suptitle = 'All specializations', colnum=6):
    """
    ---
    This function plots seaborn histograms for all data frame variables and
    saves them as .pdf and .jpg in the subfolder 'output' (pre-create it!).
    The plotting is performed on a rectangular canvas with 'colnum' columns.
    
    Parameters:
    ---
    dataframe: pandas data frame
    output_name: str (default = "histograms_all")
    The output name can also include a subfolder.
    
    suptitle: str (default = 'All specializations')

    Returns: None
    """
    rownum = ceil(dataframe.shape[1]/colnum)

    fig, axs = plt.subplots(rownum, colnum, figsize=(16, 8))
    # fig, axs = plt.subplots(rownum, colnum, figsize=(16, 8),constrained_layout = True)

    # plt.subplots_adjust(wspace=0.4)

    # set the spacing between subplots
    plt.subplots_adjust(bottom=0.1,
                        wspace=0.4,
                        hspace=0.8)
    fig.suptitle(suptitle, fontsize=15)

    for row in range(rownum):
        for col in range(colnum):
            variable_no = row*colnum+col
            if variable_no<dataframe.shape[1]:
                sns.histplot(data=dataframe,x=dataframe.columns[variable_no],
                             bins=15, stat="probability", # stat="density" # normalizing y-axis
                             kde=True, color="skyblue", ax=axs[row, col])
                axs[row, col].set(title=dataframe.columns[variable_no], xlabel=None, ylabel=None)
            else:
                axs[row, col].axis("off")

    plt.savefig("./output/"+output_name+".pdf", format="pdf", bbox_inches="tight");
    plt.savefig("./output/"+output_name+".jpg", format="jpg", bbox_inches="tight");
    plt.show()


# %%
# Joerg's visualisation function for histogram plots
def plot_histograms(df):
    for column in df.columns:
        print(column)
        plt.figure()
        plt.hist(df[column], bins=50, edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


# %%
# Adjustment of the bar width/height in seaborn.countplot:
def change_width(ax, new_width):
    for patch in ax.patches:
        current_width = patch.get_width()
        difference = current_width - new_width

        # Set new width
        patch.set_width(new_width)

        # Now Recenter the Bars
        patch.set_x(patch.get_x() + difference * .5)

def change_height(ax, new_height):
    for patch in ax.patches:
        current_height = patch.get_height()
        difference = current_height - new_height

        # Set new height
        patch.set_height(new_height)

        # Now Recenter the Bars
        patch.set_y(patch.get_y() + difference * .5)

# https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
# https://aihints.com/how-to-change-bar-width-in-seaborn/
