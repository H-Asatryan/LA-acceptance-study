#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
# # %matplotlib inline

acceptance_df = pd.read_csv("./data/clean_data/data_imputed.csv")

# +
dimension_labels = ["D", "E"]
question_labels_by_pair = [] # questions ordered subsequently
question_labels_by_dimension = [] # first desire questions, then expectation ones
for i in range(12):
    for dimension_label in dimension_labels:
        question_labels_by_pair.append(f"{i+1}-{dimension_label}")

for dimension_label in dimension_labels:
    for i in range(12):
       question_labels_by_dimension.append(f"{i+1}-{dimension_label}") 


# -

# Define the class Course
# Each instance of "Course" is a triple consisting of the
# course name, course abbreviation and course code
class Course:
    """
    Introduces courses as triples.
    
    Each instance of "Course" is a triple consisting of the
    course name, course abbreviation and course code.
    The course code is actually a list of course codes from our
    initial dataframe. This approach enables us to define so-called
    "merged courses", i.e., unions of several courses.

    Parameters:
    ---
    name: string
    token: string
    course_numbers: list of numbers
    """
    def __init__(self, name, token, course_numbers):
        self.name = name
        self.token = token
        self.course_numbers = course_numbers

# Define courses combining their names, abbreviations (for plots) and codes
Archi = Course("Architecture", "AR", [1])
Civil = Course("Civil Engineering", "CE", [2])
Electromech = Course("Electro-Mechanical Engineering", "EM", [3,5,6])
CompSci = Course("Computer Science", "CS", [4])
Sustai = Course("Sustainability", "SU", [7])
Survey = Course("Surveying", "SV", [8])
Busi = Course("Business Studies", "BS", [9])
Other = Course("Other Engineering Courses", "OE", [10])

class CourseOrganizer:
    """
    Organized course set in the following sense.
    
    For the given dataframe containing acceptance survey results
    and for the given list of "Course" instances, we filter the dataframe
    by these courses; this works for merged courses, too! Then we compute
    SELAQ feature averages for the whole dataset, as well as for the listed courses.
    From the latter we find maximum/minimum and plot them together with averages.
    The interval [min,max] is shown as an error segment. We type also course
    abbreviations on the plot to show maximizers/minimizers.

    Parameters:
    ---
    dataframe: dataframe containing acceptance survey results
    Courses: list of "Course" instances
    """
    def __init__(self, dataframe, Courses):
        self.dataframe = dataframe
        self.Courses = {}
        for Course in Courses:
            self.Courses[Course.token] = Course
        self.min_max_df = pd.DataFrame(
            columns=self.dataframe.iloc[:,5:].columns,
            index=pd.Series(["min_val", "max_val", "min_course", "max_course"])
        )
        
        self.avg_by_course = pd.DataFrame(
            columns=self.dataframe.columns,
            index=[token for token in self.Courses.keys()],
            dtype=float
        )
        
        for Course in self.Courses.values():
            course_index = pd.Series(False, index=self.dataframe.index)
            for course_num in Course.course_numbers:
                course_index = course_index | (self.dataframe["1.1"] == course_num)
            self.avg_by_course.loc[Course.token] = self.dataframe[course_index].mean()
        
        
        self.min_max_df.loc["min_val"] = self.avg_by_course.min()
        self.min_max_df.loc["min_course"] = self.avg_by_course.idxmin()
        self.min_max_df.loc["max_val"] = self.avg_by_course.max()
        self.min_max_df.loc["max_course"] = self.avg_by_course.idxmax()
        
    def plot_courses(self, filename="fig.pdf", question_labels=[]):
        
        if question_labels == []:
            question_labels = self.dataframe.iloc[:,5:].columns
        
        means = self.dataframe.iloc[:,5:].mean()
        ax = means.plot.bar(
            figsize=(10, 4),
            width=0.6,
            rot = 70,
            color="lightsteelblue" # "cadetblue", "#4F81BD"
        )
        ax.set_xticklabels(question_labels)
        
        ax.errorbar(
            question_labels, means,
            yerr=[means-self.min_max_df.loc["min_val"], self.min_max_df.loc["max_val"]-means],
            linestyle="None", marker="^", capsize=3,
#             markeredgewidth=1, # use if error bar markers are missing
            ecolor="k", fmt="k"
        )

        i = 0
        for question_label in question_labels:
            plt.text(
                question_label,
                self.min_max_df.iloc[0,i]-0.3,
                self.min_max_df.iloc[2,i],
                ha="center", va="center"
            )
            plt.text(
                question_label,
                self.min_max_df.iloc[1,i]+0.3,
                self.min_max_df.iloc[3,i],
                ha="center", va="center"
            )
            i+=1

        ax.set_ylim([0,7])
        ax.set_xlabel("Question code", size=12, labelpad=8)
        ax.set_ylabel("Average grade", size=12, labelpad=9)
        
        ax.get_figure().savefig(filename, bbox_inches='tight')
        
    def get_course_names(self):
        course_name_string = ""
        first = True
        for token, obj in self.Courses.items():
            if first == True:
                first = False
            else:
                course_name_string = f"{course_name_string}, "
            course_name_string = f"{course_name_string}{obj.name} ({token})"
        return course_name_string

CourseOrgPairwise = CourseOrganizer(
    acceptance_df, [Archi,Civil,Electromech,CompSci,Sustai,Survey,Busi,Other]
)
CourseOrgPairwise.plot_courses("./output/docs/dataset_info/"+"average_grades_specs_pairwise"+".pdf", question_labels_by_pair)


acceptance_df_resorted = pd.concat(
    [acceptance_df.iloc[:,0:5],
     acceptance_df.iloc[:,5:-1:2],
     acceptance_df.iloc[:,6:999:2]], axis=1
)
CourseOrgDimensionwise = CourseOrganizer(
    acceptance_df_resorted,
    [Archi,Civil,Electromech,CompSci,Sustai,Survey,Busi,Other]
)
CourseOrgDimensionwise.plot_courses("./output/docs/dataset_info/"+"average_grades_specs_dimensionwise"+".pdf", question_labels_by_dimension)

unimputed_index = pd.DataFrame(False, index=acceptance_df.index, columns=acceptance_df.columns[5:])
for i in range(7):
    unimputed_index = unimputed_index | (acceptance_df.iloc[:,5:] == i+1)

acceptance_df_wo_imputed = copy.deepcopy(acceptance_df)
unimputed_grades_df = acceptance_df.iloc[:,5:]
unimputed_grades_df[~unimputed_index] = np.nan
acceptance_df_wo_imputed = pd.concat([acceptance_df.iloc[:,:5], unimputed_grades_df],axis=1)

CourseOrgUnimputed = CourseOrganizer(
    acceptance_df_wo_imputed,
    [Archi,Civil,Electromech,CompSci,Sustai,Survey,Busi,Other]
)
CourseOrgUnimputed.plot_courses("./output/docs/dataset_info/"+"average_grades_specs_unimputed"+".pdf", question_labels_by_pair)
