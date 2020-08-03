"""
Viz specific util functions
"""

import matplotlib.pyplot as plt

def autolabel(rects, ax):
    """
    To add data labels for bar charts
    """
    for rect in rects:
        x = rect.get_x() + rect.get_width()/2.
        y = rect.get_height()
        
        if (y>0):
            
            ax.annotate("{}".format(round(y, 2)), (x,y), xytext=(0,5), textcoords="offset points",
                    ha='center', va='bottom')