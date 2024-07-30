# Manual clustering (tree-based approach)

# + imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.backend_bases import MouseButton

# %matplotlib widget
# -

# Import data
acceptance_df = pd.read_csv('./data/clean_data/data_imputed.csv',index_col="SurveyID")
row_mean_features_df = pd.read_csv("./data/clean_data/row_mean_features.csv",index_col="SurveyID")

# questions_complete = acceptance_df.iloc[:,4:]
# questions_df = questions_complete.loc[row_mean_features_df.index]
exp_features = ["DP Expectation", "LA General Expectation", "LA Teacher Expectation"]
questions_df = acceptance_df.iloc[:,4:]
questions_num = questions_df.shape[1] # 24
questions_df.head()

# + build tree manually
cm = mpl.cm.Set1

more_means_df = row_mean_features_df.copy()
more_means_df['Total Desire'] = row_mean_features_df.drop(columns=exp_features).mean(axis='columns')
more_means_df['Total Expectation'] = row_mean_features_df[exp_features].mean(axis='columns')

# Default (start) values for buttons (click to change)
tree_features = ['DP Desire', 'LA Desire', 'LA Expectation']
labels = np.random.randint(0, 4, size=more_means_df.shape[0])  # init labels
compare_op = ['<'] * 3

fig_tree = plt.figure(figsize=(10,3.3))

# Line plot for feature averages
ax_questions = fig_tree.add_subplot(1, 2, 1)
lines_questions = []
for i in range(4):
    # dummy plots
    x = range(questions_num)
    line, = ax_questions.plot(x, x, color=cm(i))
    lines_questions.append(line)
ax_questions.set_xlim(0, questions_num)
ax_questions.set_ylim(1, 7)
ax_questions.set_xticks(range(questions_num),minor=True)
ax_questions.grid(linestyle='dotted', linewidth=0.4, which='both')
ax_questions.set_aspect(3)
ax_questions.set_title("Average grades per question", pad=10)

# 3D scatter plot for clusters
ax_feat_space = fig_tree.add_subplot(1, 2, 2, projection='3d')
sc_feat_space = ax_feat_space.scatter(*more_means_df[tree_features].T.to_numpy())
ax_feat_space.set_xlabel(tree_features[0])
ax_feat_space.set_ylabel(tree_features[1])
ax_feat_space.set_zlabel(tree_features[2])
ax_feat_space.set_title("Clusters", pad=0)

fig_tree.subplots_adjust(left=0.35, top=0.99, bottom=0.05, right=0.915)

# Plot buttons and sliders
ax_drawing = fig_tree.add_axes([0.03, 0, 0.25, 1])
ax_drawing.set_xlim(0, 1)
ax_drawing.set_ylim(0, 1)
ax_drawing.axis('off')

# Value sliders
ax_slider = [fig_tree.add_axes([0.04, 0.9 - i * 0.33, 0.19, 0.02]) for i in range(3)]
percentile_slider = [Slider(ax=ax_slider[i], label="", orientation="horizontal",
                            valmin=0.0, valmax=1, valinit=0.5) for i in range(3)]

# Buttons with feature text
ax_button = [fig_tree.add_axes([0.04, 0.93 - i * 0.33, 0.19, 0.06]) for i in range(3)]
feat_button = [Button(ax=ax_button[i], label=tree_features[i]) for i in range(3)]

ax_invert_button = [fig_tree.add_axes([0.085, 0.8 - i * 0.33, 0.09, 0.05]) for i in range(3)]
invert_button = [Button(ax=ax_invert_button[i], label='← yes', color='w') for i in range(3)]

# Down arrows at the left side
for i in range(2):
    ax_drawing.arrow(0.1, 0.9 - i * 0.33, 0.0, -0.235, width=0.03,
                     head_length=0.03, length_includes_head=True, fc='gray')
ax_drawing.arrow(0.1, 0.9 - 2 * 0.33, 0.0, -0.115, width=0.03,
                 head_length=0.03, length_includes_head=True, fc='gray')

# Bullet at the text label
circle = mpl.patches.Ellipse((0.1, 0.76 - 2 * 0.33), 0.05, 0.025, color=cm(3))
ax_drawing.add_patch(circle)

# Down arrows at the right side
for i in range(3):
    ax_drawing.arrow(0.7, 0.9 - i * 0.33, 0.0, -0.115, width=0.03, head_length=0.03,
                     length_includes_head=True, fc='gray')
    # Bullets at text labels
    circle = mpl.patches.Ellipse((0.7, 0.76 - i * 0.33), 0.05, 0.025, color=cm(i))
    ax_drawing.add_patch(circle)

# Text labels with percentage
size_labels = [ax_drawing.text(0.65, 0.7 - i * 0.33, '%') for i in range(3)]
size_labels.append(ax_drawing.text(0.05, 0.7 - 2 * 0.33, '%'))

# Text labels with limit value
lim_labels = [ax_drawing.text(0.8, 0.77 - i * 0.33, '-') for i in range(3)]

def update_labels(val=None):
    labels[:] = 3
    for i in range(3):
        feat = more_means_df[tree_features[i]]
        lim = feat.quantile(percentile_slider[i].val)
        idx = (feat <= lim) if compare_op[i] == '<' else (feat >= lim)
        idx &= labels == 3  # not yet assigned
        labels[idx] = i
        lim_labels[i].set_text(f'Threshold:\n{lim:.2f}')

    for i in range(4):
        idx = labels == i
        question_means = questions_df[idx].mean()
        lines_questions[i].set_ydata(question_means)

        size_labels[i].set_text(f'{idx.mean()*100:.1f}%')

    sc_feat_space.set_facecolor(cm(labels))

def update_invert_fact(i):
    def update_invert_button(evt):
        compare_op[i] = '<' if compare_op[i] != '<' else '>'
        invert_button[i].label.set_text('← yes' if compare_op[i] == '<' else 'yes →')

        update_labels()

    return update_invert_button


def update_features():
    sc_feat_space.set_offsets(more_means_df[tree_features])

    update_labels()

    fig_tree.canvas.draw()
    fig_tree.canvas.flush_events()


def update_feat_fact(i):
    def update_button_feature(evt):
        # select next feature
        feat = tree_features[i]
        idx = more_means_df.columns.get_loc(feat)
        new_idx = (idx + (+1 if evt.button == MouseButton.LEFT else -1)) % more_means_df.columns.shape[0]
        new_feat = more_means_df.columns[new_idx]
        tree_features[i] = new_feat
        feat_button[i].label.set_text(new_feat)

        if i == 0: ax_feat_space.set_xlabel(new_feat)
        if i == 1: ax_feat_space.set_ylabel(new_feat)
        if i == 2: ax_feat_space.set_zlabel(new_feat)

        update_features()

    return update_button_feature


for i in range(3):
    percentile_slider[i].on_changed(update_labels)
    invert_button[i].on_clicked(update_invert_fact(i))
    feat_button[i].on_clicked(update_feat_fact(i))

update_labels()

plt.savefig("./output/clustering/manual/manual_clusters_v1"+".jpg",
            format="jpg", dpi=200, bbox_inches="tight");
