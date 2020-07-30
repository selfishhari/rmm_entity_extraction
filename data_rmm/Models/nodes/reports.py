""" Generate visualizations"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from nodes import vis_utils
from sklearn.metrics import accuracy_score
import math
import os
import seaborn as sns




################### Get resolution time ###################

def _get_time_diff(x):
    """
    Helper function to get time difference
    """
    
    x_dt = pd.to_datetime(x, format="%H:%M:%S")
    
    min_time = min(x_dt)
    
    max_time = max(x_dt)
    
    return (max_time - min_time).seconds/60

def get_conversation_duration(data: pd.DataFrame, collated_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time taken(mins) for resolution per session_id
    """
    
    conv_time = data.groupby("session_id")["time"].apply(_get_time_diff).reset_index()
    
    collated_data = collated_data.merge(conv_time, on="session_id", how="left")
    
    collated_data.rename({"time":"conv_duration"}, axis=1, inplace=True)
    
    return collated_data

    
def bin_conv_duration(data: pd.DataFrame) -> pd.DataFrame:
    """
    bins conversation duration
    """
    
    # intervals of 5 from 0 to 50 then last bin of 200
    time_bins = [x for x in range(-5, 55, 5)] + [200]
    
    #time_bin_labels = ["<5"] + [x for x in range(5, 55, 5)] #+ [">55"]
    
    data["conv_duration_bins"] = pd.cut(data["conv_duration"], time_bins, right=True)
    
    
    return data

def _group_intents(parameters:Dict, x) -> pd.DataFrame:
    
    if x in parameters["claim_intents"]:
        
        return "claim_intents"
    
    elif x in parameters["policy_intents"]:
        
        return "policy_intents"
    
    else:
        
        return "rest"

def get_duration_count_report_data(data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    
    data["grouped_intents"] = data["label"].apply(lambda x: _group_intents(parameters, x))
    
    agg_durations = data.groupby(["grouped_intents", "conv_duration_bins"])["session_id"].nunique().reset_index(name="count_sessions")

    agg_durations["group_conv_counts"] = agg_durations.groupby("grouped_intents")["count_sessions"].transform("sum")
    
    total_convs = agg_durations.loc[:, "count_sessions"].sum()
    
    agg_durations["% Conversations"] = agg_durations["count_sessions"] / total_convs *100
    
    agg_durations["% Conversations(% of Intent Group)"] = agg_durations["count_sessions"] / agg_durations["group_conv_counts"] *100
    
    return agg_durations

def _get_master_table(unlabelled_data: pd.DataFrame, collated_data: pd.DataFrame):
    
    master_data = unlabelled_data.merge(collated_data, on="session_id", how="left")
    
    conv_time = master_data.groupby("session_id")["time"].apply(_get_time_diff).reset_index()
    
    conv_time.rename({"time":"conv_duration"}, axis=1, inplace=True)
    
    master_data = master_data.merge(conv_time, on="session_id", how="left")
    
    master_data["date"] = pd.to_datetime(master_data["start_time"])
    
    master_data["quarter"] = master_data["date"].dt.quarter
    
    master_data["year"] = master_data["date"].dt.year
    
    master_data["month"] = master_data["date"].dt.month
    
    return master_data

def get_time_series_report_data(unlabelled_data: pd.DataFrame, collated_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    This creates year-monthly statistics
    """
    
    master_data = _get_master_table(unlabelled_data, collated_data)
    
    master_data["year_month"] = master_data["date"].apply(lambda x: x.strftime("%Y_%m"))
    
    master_data["year_quarter"] = master_data["year"].astype(str) + "_" + master_data["quarter"].astype(str)
    
    master_data["grouped_intents"] = master_data["label"].apply(lambda x: _group_intents(parameters, x))
    
    conv_binned_data = bin_conv_duration(master_data)

    ts_data = conv_binned_data.groupby(["grouped_intents", "year_month"]).agg({"session_id":"nunique",
                                                                                  "conv_duration":["mean", "sum"]
                                                                                  }).reset_index()
    
    ts_data.columns = [''.join(col) if col[1] == '' else '_'.join(col) for col in ts_data.columns]
    
    ts_data = ts_data.rename_axis(None, axis=1)
    
    ts_data["grouped_session_counts"] = ts_data.groupby(["grouped_intents"])["session_id_nunique"].transform("sum")
    
    ts_data["% Conversations(% of Intent Group Totals)"] = ts_data["session_id_nunique"] / ts_data["grouped_session_counts"] * 100
    
    ts_data["% Conversations"] = ts_data["session_id_nunique"] / ts_data["session_id_nunique"].sum() * 100
    
    ts_data = ts_data.merge(master_data[["year_month", "quarter", "year", "year_quarter"]].drop_duplicates(), on="year_month", how="left")
    
    
    
    #Quarterly report
    
    ts_data_quarter = conv_binned_data.groupby(["grouped_intents", "year_quarter"]).agg({"session_id":"nunique",
                                                                                  "conv_duration":["mean", "sum"]
                                                                                  }).reset_index()
    
    ts_data_quarter.columns = [''.join(col) if col[1] == '' else '_'.join(col) for col in ts_data_quarter.columns]
    
    ts_data_quarter = ts_data_quarter.rename_axis(None, axis=1)
    
    ts_data_quarter["grouped_session_counts"] = ts_data_quarter.groupby(["grouped_intents"])["session_id_nunique"].transform("sum")
    
    ts_data_quarter["% Conversations(% of Intent Group Totals)"] = ts_data_quarter["session_id_nunique"] / ts_data_quarter["grouped_session_counts"] * 100
    
    ts_data_quarter["% Conversations"] = ts_data_quarter["session_id_nunique"] / ts_data_quarter["session_id_nunique"].sum() * 100
    
    return [ts_data ,ts_data_quarter]

def get_time_series_report_data_ungrouped(unlabelled_data: pd.DataFrame, collated_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    This creates year-monthly statistics
    """
    
    master_data = _get_master_table(unlabelled_data, collated_data)
    
    master_data["year_month"] = master_data["date"].apply(lambda x: x.strftime("%Y_%m"))
    
    master_data["year_quarter"] = master_data["year"].astype(str) + "_" + master_data["quarter"].astype(str)
    
    master_data["grouped_intents"] = master_data["label"]#.apply(lambda x: _group_intents(parameters, x))
    
    conv_binned_data = bin_conv_duration(master_data)

    ts_data = conv_binned_data.groupby(["grouped_intents", "year_month"]).agg({"session_id":"nunique",
                                                                                  "conv_duration":["mean", "sum"]
                                                                                  }).reset_index()
    
    ts_data.columns = [''.join(col) if col[1] == '' else '_'.join(col) for col in ts_data.columns]
    
    ts_data = ts_data.rename_axis(None, axis=1)
    
    ts_data["grouped_session_counts"] = ts_data.groupby(["grouped_intents"])["session_id_nunique"].transform("sum")
    
    ts_data["% Conversations(% of Intent Group Totals)"] = ts_data["session_id_nunique"] / ts_data["grouped_session_counts"] * 100
    
    ts_data["% Conversations"] = ts_data["session_id_nunique"] / ts_data["session_id_nunique"].sum() * 100
    
    ts_data = ts_data.merge(master_data[["year_month", "quarter", "year", "year_quarter"]].drop_duplicates(), on="year_month", how="left")
    
    
    
    #Quarterly report
    
    ts_data_quarter = conv_binned_data.groupby(["grouped_intents", "year_quarter"]).agg({"session_id":"nunique",
                                                                                  "conv_duration":["mean", "sum"]
                                                                                  }).reset_index()
    
    ts_data_quarter.columns = [''.join(col) if col[1] == '' else '_'.join(col) for col in ts_data_quarter.columns]
    
    ts_data_quarter = ts_data_quarter.rename_axis(None, axis=1)
    
    ts_data_quarter["grouped_session_counts"] = ts_data_quarter.groupby(["grouped_intents"])["session_id_nunique"].transform("sum")
    
    ts_data_quarter["% Conversations(% of Intent Group Totals)"] = ts_data_quarter["session_id_nunique"] / ts_data_quarter["grouped_session_counts"] * 100
    
    ts_data_quarter["% Conversations"] = ts_data_quarter["session_id_nunique"] / ts_data_quarter["session_id_nunique"].sum() * 100
    
    return [ts_data ,ts_data_quarter]



def plot_conv_duration_counts(data: pd.DataFrame):
    
    plt.rcParams["figure.figsize"] = [16,9]
    
    #Plot 1
    
    duration_counts_plt = data.pivot(index="conv_duration_bins",
                                     columns="grouped_intents",
                                     values="% Conversations").plot.bar()
    
    vis_utils.autolabel(duration_counts_plt.patches, duration_counts_plt)
    
    plt.ylabel("% Conversations")
    
    plt.xlabel("Conversation Duration(mins)")
    
    plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_perc_conversations.png") )
    
    # Plot 2
    
    duration_counts_plt_v2 = data.pivot(index="conv_duration_bins",
                                        columns="grouped_intents",
                                        values="count_sessions").plot.bar()
    
    vis_utils.autolabel(duration_counts_plt_v2.patches, duration_counts_plt_v2)
    
    plt.ylabel("# Conversations")
    
    plt.xlabel("Conversation Duration(mins)")
    
    plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_count_conversations.png"))
    
    # Plot 3- As % of grouped intents
    duration_counts_plt_v3 = data.pivot(index="conv_duration_bins",
                                        columns="grouped_intents",
                                        values="% Conversations(% of Intent Group)").plot.bar(width=0.3)
    
    vis_utils.autolabel(duration_counts_plt_v3.patches, duration_counts_plt_v3)
    
    plt.ylabel("% Conversations")
    
    plt.xlabel("Conversation Duration(mins)")
    
    plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_grouped_perc_conversations.png"))
    
    return

def plot_time_series_trends(data: pd.DataFrame, data_quarterly):
    """
    Creates time series plots
    """
    
    plt.rcParams["figure.figsize"] = [12,9]
    
    #Plot 1
    
    ts_plot1 = data.pivot(index="year_month", 
                                     columns="grouped_intents",
                                     values="% Conversations").plot.line()
    vis_utils.autolabel(ts_plot1.patches, ts_plot1)
    
    plt.ylabel("% Conversations")
    
    plt.xlabel("Year-Month")
    
    #plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_perc_conversations.png") )
    
    # Plot 2
    
    ts_plot2 = data.pivot(index="year_month", 
                                     columns="grouped_intents",
                                     values="% Conversations(% of Intent Group Totals)").plot.line()
    vis_utils.autolabel(ts_plot2.patches, ts_plot2)
    
    plt.ylabel("% Conversations(% of Intent Group Totals)")
    
    plt.xlabel("Year-Month")
    
    #plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_count_conversations.png"))
    
    
    
    # Plot 3- As % of grouped intents
    
    ts_plot3 = data.pivot(index="year_month", 
                                     columns="grouped_intents",
                                     values="conv_duration_mean").plot.line()
    vis_utils.autolabel(ts_plot3.patches, ts_plot3)
    
    plt.ylabel("Average Conversation Duration(Mins)")
    
    plt.xlabel("Year-Month")
    
    #plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_grouped_perc_conversations.png"))
    
    ts_plot4 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="session_id_nunique", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("No. Of Conversations")
    
    plt.xlabel("Year-Quarter")
    
    # Plot 5
    
    ts_plot5 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="% Conversations", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("% Conversations")
    
    plt.xlabel("Year-Quarter")
    
    ts_plot6 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="% Conversations(% of Intent Group Totals)", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("% Conversations(% of Intent Groups)")
    
    plt.xlabel("Year-Quarter")
    
    ts_plot6 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="conv_duration_mean", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("Average Conversation Duration(mins)")
    
    plt.xlabel("Year-Quarter")
    
    return

def plot_classification_time_series_trends(data: pd.DataFrame, data_quarterly, data_sub_monthly, data_sub_quarterly):
    """
    Creates time series plots
    """
    
    plt.rcParams["figure.figsize"] = [12,9]
    
    data_sub_monthly["grouped_intents"] = data_sub_monthly["grouped_intents"].apply(lambda x: str(x)+"(not automated)")
    
    data = pd.concat([data, data_sub_monthly], axis=0, ignore_index=True)
    
    data_quarterly = pd.concat([data_quarterly, data_sub_quarterly], axis=0, ignore_index=True)
    
    #Plot 1
    
    ts_plot1 = data.pivot(index="year_month", 
                                     columns="grouped_intents",
                                     values="% Conversations").plot.line()
    vis_utils.autolabel(ts_plot1.patches, ts_plot1)
    
    plt.ylabel("% Conversations")
    
    plt.xlabel("Year-Month")
    
    #plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_perc_conversations.png") )
    
    # Plot 2
    
    ts_plot2 = data.pivot(index="year_month", 
                                     columns="grouped_intents",
                                     values="% Conversations(% of Intent Group Totals)").plot.line()
    vis_utils.autolabel(ts_plot2.patches, ts_plot2)
    
    plt.ylabel("% Conversations(% of Intent Group Totals)")
    
    plt.xlabel("Year-Month")
    
    #plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_count_conversations.png"))
    
    
    
    # Plot 3- As % of grouped intents
    
    ts_plot3 = data.pivot(index="year_month", 
                                     columns="grouped_intents",
                                     values="conv_duration_mean").plot.line()
    vis_utils.autolabel(ts_plot3.patches, ts_plot3)
    
    plt.ylabel("Average Conversation Duration(Mins)")
    
    plt.xlabel("Year-Month")
    
    #plt.savefig(os.path.join("data", "08_reporting", "conversation_duration_vs_grouped_perc_conversations.png"))
    
    ts_plot4 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="session_id_nunique", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("No. Of Conversations")
    
    plt.xlabel("Year-Quarter")
    
    # Plot 5
    
    ts_plot5 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="% Conversations", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("% Conversations")
    
    plt.xlabel("Year-Quarter")
    
    ts_plot6 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="% Conversations(% of Intent Group Totals)", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("% Conversations(% of Intent Groups)")
    
    plt.xlabel("Year-Quarter")
    
    ts_plot6 = data_quarterly.pivot_table(index="year_quarter", 
                                     columns="grouped_intents",
                                     values="conv_duration_mean", aggfunc='sum').plot.line()
    #vis_utils.autolabel(ts_plot4.patches, ts_plot4)
    
    plt.ylabel("Average Conversation Duration(mins)")
    
    plt.xlabel("Year-Quarter")
    
    return





######################### Accuracy - Population Report ####################

def get_accuracy_pop_data(cm_data: pd.DataFrame, group_tail_labels: bool) -> pd.DataFrame:
    
    cm_data = cm_data.copy()
    
    col_names = cm_data.columns.tolist()
    
    col_names[0] = "labels"

    cm_data.columns = col_names

    cm_data.loc[:20,:]

    total_data = cm_data.loc[:20,"actuals"].sum()

    cm_data["actuals_%"] = cm_data["actuals"] / total_data *100
    
    cm_precision = cm_data.loc[(cm_data.labels=="precision") | \
                           (cm_data.labels=="predicted"), :].T.reset_index()
    
    cm_precision.columns = ["labels", "predicted", "precision"]

    precision_pop_data = cm_precision.merge(cm_data[["labels", "actuals_%", "actuals"]], on="labels")

    precision_pop_data.sort_values("precision", ascending=False, inplace=True)
    
    precision_pop_data.reset_index(drop=True, inplace=True)
    
    precision_pop_data["precision"] = precision_pop_data["precision"]*100
    
    precision_pop_data["predicted_correct"] = (precision_pop_data["predicted"] * precision_pop_data["precision"].fillna(0) / 100).astype(int)
    
    if group_tail_labels:
        
        last_indices = precision_pop_data.loc[10:, :].index.tolist()
        
        oth_ind = precision_pop_data.loc[precision_pop_data.labels == "others",:].index.tolist()
        
        last_indices = last_indices + oth_ind
        
        precision_pop_data.loc[last_indices, "actuals"] = precision_pop_data.loc[last_indices, "actuals"].sum()
        
        precision_pop_data.loc[last_indices, "predicted_correct"] = precision_pop_data.loc[last_indices, "predicted_correct"].sum()
        
        precision_pop_data.loc[last_indices, "predicted"] = precision_pop_data.loc[last_indices, "predicted"].sum()
        
        precision_pop_data.loc[last_indices, "precision"] = precision_pop_data.loc[last_indices, "predicted_correct"] / precision_pop_data.loc[last_indices, "predicted"] * 100
        
        precision_pop_data.loc[last_indices, "actuals_%"] = precision_pop_data.loc[last_indices, "actuals"] / total_data *100
        
        precision_pop_data.loc[last_indices, "labels"] = "all_other_classes" 
        
        precision_pop_data = precision_pop_data.drop(last_indices[1:])
        
        precision_pop_data.sort_values("precision", ascending=False, inplace=True)
    
        precision_pop_data.reset_index(drop=True, inplace=True)
        
    precision_pop_data["population_%"] = precision_pop_data["actuals_%"].cumsum()

    precision_pop_data["predicted_correct_cumsum"] = precision_pop_data["predicted_correct"].cumsum()

    precision_pop_data["predicted_all_cumsum"] = precision_pop_data["predicted"].cumsum()

    precision_pop_data["accuracy_%"] = precision_pop_data["predicted_correct_cumsum"] / precision_pop_data["predicted_all_cumsum"] * 100

    
    return precision_pop_data


def plot_accuracy_pop_chart(precision_pop_data: pd.DataFrame, name_suffix=""):
    
    plt.rcParams["figure.figsize"] = [16,9]
    
    precision_pop_data = precision_pop_data.copy()
    
    precision_pop_data_plt = precision_pop_data[["labels", "population_%"]].rename({"population_%":"% Population (cumulative)"}, axis = 1 ).plot.bar(color="#b2c9ed")

    precision_pop_data_plt.set_ylim(ymax=300, ymin=0)

    precision_pop_data_plt.set_ylabel("")

    precision_pop_data_plt.set_xlabel("Labels")

    precision_pop_data_plt.legend(loc="upper left")

    precision_pop_data_plt.set_yticks([])

    vis_utils.autolabel(precision_pop_data_plt.patches, precision_pop_data_plt)

    plt.xticks(precision_pop_data.index, precision_pop_data["labels"].values)

    ax = plt.twinx()

    ax.plot(precision_pop_data["accuracy_%"], color="g")

    for i,j in precision_pop_data["accuracy_%"].items():
        ax.annotate(str(round(j, 1)), xy=(i, j+1))

    #axes2.set_ylim(-1, 1)
    ax.set_ylabel('')

    ax.legend(['% Accuracy (cumulative)'], loc="upper left", bbox_to_anchor=(0, 0.9))

    ax.set_ylim(ymin=10, ymax=120)

    ax.set_yticks([])


    plt.savefig(os.path.join("data", "08_reporting", "accuracy_pop_chart_cum_chart_"+ name_suffix +".png"), bbox_inches="tight")
    
    return

def plot_accuracy_pop_noncum_chart(precision_pop_data: pd.DataFrame, name_suffix=""):
    
    plt.rcParams["figure.figsize"] = [16,9]
    
    precision_pop_data = precision_pop_data.copy()
    
    precision_pop_data_plt = precision_pop_data[["labels", "actuals_%"]].rename({"actuals_%":"% Population"}, axis = 1 ).plot.bar(color="#b2c9ed")

    precision_pop_data_plt.set_ylim(ymax=90, ymin=0)

    precision_pop_data_plt.set_ylabel("")

    precision_pop_data_plt.set_xlabel("Labels")

    precision_pop_data_plt.legend(loc="upper left", bbox_to_anchor=(0, 0.9))

    precision_pop_data_plt.set_yticks([])

    vis_utils.autolabel(precision_pop_data_plt.patches, precision_pop_data_plt)

    plt.xticks(precision_pop_data.index, precision_pop_data["labels"].values)

    ax = plt.twinx()

    ax.plot(precision_pop_data["precision"], color="g")

    for i,j in precision_pop_data["precision"].items():
        ax.annotate(str(round(j, 1)), xy=(i, j+1))

    #axes2.set_ylim(-1, 1)
    ax.set_ylabel('')

    ax.legend(['% Accuracy'], loc="upper left", bbox_to_anchor=(0, 1))

    ax.set_ylim(ymin=-50, ymax=150)

    ax.set_yticks([])

    plt.savefig(os.path.join("/home/viky/Desktop/freelancer/NB-Custom/reports/"+ "08_reporting", "accuracy_pop_chart_cum_chart_"+ name_suffix +".png"), bbox_inches="tight")
    
    return

def _get_decile_accuracy(df):
    
    acc = accuracy_score(df["label_enc"], df["pred_label_enc"])
    
    return acc

def _get_decile_error_rate(df):
    
    acc = accuracy_score(df["label_enc"], df["pred_label_enc"])
    
    return (1-acc)

def _get_hit_rate(df):
    
    total_corrects = df["label_enc"].sum()
    
    total_records = df.shape[0]
    
    hit_rate = total_corrects / total_records
    
    return hit_rate


def get_combined_error_rates_data(data: pd.DataFrame) -> pd.DataFrame:
    
    data = data.sort_values("pred_label_prob")

    data["index_nums"] = data.reset_index(drop=True).index.tolist()

    data["quintiles"] = data["index_nums"] // (math.ceil(data.shape[0]/5))

    data["deciles"] = data["index_nums"] // (math.ceil(data.shape[0]/10))

    data["5_percentiles"] = data["index_nums"] // (math.ceil(data.shape[0]/20))
    
    #percentile_5_hitrate = data.groupby("5_percentiles").apply(_get_hit_rate).reset_index(name="hit_rate")

    percentile_5_error_rate = data.groupby("5_percentiles").apply(_get_decile_error_rate).reset_index(name="error_rate")

    #percentile_5_data = percentile_5_accuracy.merge(percentile_5_error_rate)
    
    #decile_hitrate = data.groupby("decile").apply(_get_hit_rate).reset_index(name="hit_rate")

    decile_error_rate = data.groupby("deciles").apply(_get_decile_error_rate).reset_index(name="error_rate")

    #decile_data = decile_accuracy.merge(percentile_5_error_rate)
    
    quintile_error_rate = data.groupby("quintiles").apply(_get_decile_error_rate).reset_index(name="error_rate")

    return [decile_error_rate, percentile_5_error_rate, quintile_error_rate, data]

def plot_error_rates(data, overall_error_rate):
    """
    Plot error rate bar graphs
    """
    
    plt.rcParams["figure.figsize"] = [16,9]
    
    plot = (data["error_rate"]* 100).plot.bar(color="#b2c9ed")

    plot.axhline(y=overall_error_rate  * 100, color="red", linestyle="--")
    
    vis_utils.autolabel(plot.patches, plot)
    
    plot.set_xlabel("Deciles")
    
    plot.set_ylabel("Error Rate(%)")
    
    plot.legend(['Overall Error Rate'], loc="upper right")#, bbox_to_anchor=(0, 1))
    
    y_ticks = sorted([10, 20, 23.2, 30, 40, 50, 60, 70, 80, 90, 100, overall_error_rate * 100] )
    
    plot.set_yticks(y_ticks)
    
    plt.savefig(os.path.join("/home/viky/Desktop/freelancer/NB-Custom/reports", "08_reporting", "error_rates_chart.png"), bbox_inches="tight")
    
    plot.set_title("Decile Error Rates")
    
    return

def plot_deciles_breakdown(test_pred, test_pred_orig_label_counts):
    
    decile_label_counts = test_pred.groupby("deciles").agg(label_counts=("pred_label","value_counts")).reset_index()

    decile_label_counts = test_pred.groupby("deciles").agg(sessions=("session_id","nunique")).reset_index().merge(decile_label_counts, on="deciles")

    decile_label_counts["perc_labels"] = decile_label_counts["label_counts"] / decile_label_counts["sessions"]

    decile_label_counts_pivots = decile_label_counts.pivot(index="deciles", columns="pred_label", values="perc_labels").reset_index()

    colors_list = ['#2A27CB',
     '#ABD234',
     '#BBBBBB',
     '#6117ED',
     '#9EFEE3',
     '#4589B3',
     '#A2807F',
     '#A312DE',
     '#F5420A',
     '#2F817B',
     '#052D79',
     '#1DDCD2',
     '#D2CFE9',
     '#106EB0',
     '#84A9E1',
     '#BB9B3C',
     '#DBE0BD',
     '#4CCF1E',
     '#D58D85',
     '#C524B2',
     '#000000']

    sns.set()
    
    decile_label_counts_pivots["deciles"] = decile_label_counts_pivots["deciles"] + 1
    
    decile_labels_plot = decile_label_counts_pivots.set_index('deciles').plot(kind='bar', stacked=True, color=colors_list)

    #vis_utils.autolabel(plot.patches, plot)

    decile_labels_plot.legend(loc="upper left")

    decile_labels_plot.legend(loc="upper left", bbox_to_anchor=(-0.3,1))
    
    decile_labels_plot.set_title("Breakdown of intents")
    
    
    
    #decile_labels_plot.show()
    
    ###########################################################################################
    
    test_pred["correct"] = test_pred["label"] == test_pred["pred_label"]

    test_pred["cum_corrects"] = test_pred.sort_values("pred_label_prob", ascending=False)["correct"].cumsum()

    test_pred["cum_convs_count"] = test_pred.reset_index().sort_values("pred_label_prob", ascending=False).index + 1

    test_pred["cum_accuracy"] = test_pred["cum_corrects"] / test_pred["cum_convs_count"]

    conv_counts_above_95 = sum(test_pred.loc[:,"cum_accuracy"] >= 0.95)

    print("Percentage population above 95% accuracy", conv_counts_above_95 / 1837)

    top_preds = test_pred.loc[test_pred["cum_accuracy"] >= 0.95, :]

    top_preds_counts = top_preds.groupby("pred_label").size().reset_index(name="label_counts")

    top_preds_counts["label_perc"] = top_preds_counts["label_counts"] / conv_counts_above_95

    top_preds_counts_plot = top_preds_counts[["pred_label", "label_perc"]].sort_values("label_perc", ascending=False).set_index("pred_label").plot(kind="bar", color="#b2c9ed")

    vis_utils.autolabel(top_preds_counts_plot.patches, top_preds_counts_plot)
    
    top_preds_counts_plot.set_xlabel("Intents")
    
    top_preds_counts_plot.set_ylabel("Population %")
    
    top_preds_counts_plot.set_yticks([])
    
    top_preds_counts_plot.set_title("Breakdown of 95%+ confident classifications")
    
    #top_preds_counts_plot.show()
    
    
    ################################################################################
    
    top_3_deciles = test_pred.loc[test_pred["deciles"] >= 5, :]
    
    conv_counts_top3 = sum(test_pred.loc[:,"deciles"] >= 5)

    top_3_dec_counts = top_3_deciles.groupby("pred_label").size().reset_index(name="label_counts")
    
    top_3_dec_counts["label_perc"] = top_3_dec_counts["label_counts"] / conv_counts_top3

    top_3_dec_counts_plot = top_3_dec_counts[["pred_label", "label_perc"]].sort_values("label_perc", ascending=False).set_index("pred_label").plot(kind="bar", color="#b2c9ed")

    vis_utils.autolabel(top_3_dec_counts_plot.patches, top_3_dec_counts_plot)
    
    top_3_dec_counts_plot.set_xlabel("Intents")
    
    top_3_dec_counts_plot.set_ylabel("Population %")
    
    top_3_dec_counts_plot.set_yticks([])
    
    top_3_dec_counts_plot.set_title("Breakdown of top 3 deciles")
    
    ############################# recall_plot ##################################################
    
    
    
    test_pred_tp = test_pred.loc[test_pred["label"] == test_pred["pred_label"], : ]
    
    test_pred_tp_counts = test_pred_tp.groupby(["pred_label", "deciles"]).size().reset_index(name="label_counts")
    
    recall_data = test_pred_orig_label_counts.merge(test_pred_tp_counts, left_on=["label"], right_on=["pred_label"], how="right")
    
    recall_data["label_recall_perc"] = recall_data["label_counts"] / recall_data["orig_label_counts"]
    
    #top_3_dec_recall_plot = recall_data[["label", "label_recall_perc"]].sort_values("label_recall_perc", ascending=False).set_index("label").plot(kind="bar", color="#b2c9ed")
    
    top_3_dec_recall_plot = recall_data.pivot(index="deciles",
                                     columns="label",
                                     values="label_recall_perc").plot.bar()
    
    vis_utils.autolabel(top_3_dec_recall_plot.patches, top_3_dec_recall_plot)
    
    top_3_dec_recall_plot.set_xlabel("Decile - Intents")
    
    top_3_dec_recall_plot.set_ylabel("Population %")
    
    top_3_dec_recall_plot.set_yticks([])
    
    top_3_dec_recall_plot.set_title("Recall breakdown of deciles")
    
    recall_data.sort_values(["label", "deciles"], ascending=False, inplace = True)

    recall_data["cum_label_counts"] = recall_data.groupby(["label"])["label_counts"].transform("cumsum")

    recall_data["cum_recall"] = recall_data["cum_label_counts"] / recall_data["orig_label_counts"]

    top_3_dec_cum_recall_plot = recall_data.pivot(index="deciles",
                                     columns="label",
                                     values="cum_recall").plot.bar()
    
    vis_utils.autolabel(top_3_dec_cum_recall_plot.patches, top_3_dec_cum_recall_plot)
    
    top_3_dec_cum_recall_plot.set_xlabel("Decile - Intents")
    
    top_3_dec_cum_recall_plot.set_ylabel("Population %")
    
    top_3_dec_cum_recall_plot.set_yticks([])
    
    top_3_dec_cum_recall_plot.set_title("Cumulative Recall breakdown of deciles")
    
    return [test_pred, decile_label_counts_pivots, top_preds_counts, conv_counts_above_95, conv_counts_above_95 / test_pred.shape[0], recall_data]


def get_automated_ts_data_splits(unlabelled_raw, collated_data, test_pred, test_pred_breakdowns, parameters):
    
    """
    Returns collated time series data for monthly and quartly level with automated and unautomated splits
    """

    unlabelled_raw_test = unlabelled_raw.loc[unlabelled_raw.session_id.isin(test_pred.session_id), :]

    collated_data_test = collated_data.loc[collated_data.session_id.isin(test_pred.session_id), :]

    automated_sid = test_pred_breakdowns.loc[(test_pred_breakdowns.cum_accuracy >= 0.95) & \
                                          ((test_pred_breakdowns["label"] == test_pred_breakdowns["pred_label"])), :].session_id

    unautomated_sid = test_pred_breakdowns.loc[(test_pred_breakdowns.cum_accuracy < 0.95) | \
                                            (test_pred_breakdowns["label"] != test_pred_breakdowns["pred_label"]), :].session_id

    unlabelled_raw_minus_top3_ug = unlabelled_raw.loc[(unlabelled_raw.session_id.isin(unautomated_sid)), :]

    collated_data_minus_top3_ug = collated_data.loc[(collated_data.session_id.isin(unautomated_sid)), :]


    unlabelled_raw_top3_ug = unlabelled_raw.loc[(unlabelled_raw.session_id.isin(automated_sid)), :]

    collated_data_top3_ug = collated_data.loc[(collated_data.session_id.isin(automated_sid)), :]

    
    ts_data_ungrouped_monthly, ts_data_ungrouped_quarterly = get_time_series_report_data_ungrouped(unlabelled_raw_test, collated_data_test, parameters)

    

    ts_data_ungrouped_monthly_top3, ts_data_ungrouped_quarterly_top3 = get_time_series_report_data_ungrouped(unlabelled_raw_top3_ug, collated_data_top3_ug, parameters)

    

    ts_data_ungrouped_monthly_minus_top3, ts_data_ungrouped_quarterly_minus_top3 = get_time_series_report_data_ungrouped(unlabelled_raw_minus_top3_ug, collated_data_minus_top3_ug, parameters)

    ts_data_ungrouped_monthly_minus_top3["grouped_intents"] = ts_data_ungrouped_monthly_minus_top3["grouped_intents"].apply(lambda x: str(x)+"(not automated)")

    ts_data_ungrouped_monthly_top3["grouped_intents"] = ts_data_ungrouped_monthly_top3["grouped_intents"].apply(lambda x: str(x)+"(automated)")

    ts_data_ungrouped_monthly_all = pd.concat([ts_data_ungrouped_monthly, ts_data_ungrouped_monthly_minus_top3, ts_data_ungrouped_monthly_top3], axis=0, ignore_index=True)

    ts_data_ungrouped_quarterly_minus_top3["grouped_intents"] = ts_data_ungrouped_quarterly_minus_top3["grouped_intents"].apply(lambda x: str(x)+"(not automated)")

    ts_data_ungrouped_quarterly_top3["grouped_intents"] = ts_data_ungrouped_quarterly_top3["grouped_intents"].apply(lambda x: str(x)+"(automated)")

    ts_data_ungrouped_quarterly_all = pd.concat([ts_data_ungrouped_quarterly, ts_data_ungrouped_quarterly_minus_top3, ts_data_ungrouped_quarterly_top3], axis=0, ignore_index=True)

    return ts_data_ungrouped_quarterly_all, ts_data_ungrouped_monthly_all





######################### Redirection Report ##############################

def get_time_to_redirection(unlabelled_data: pd.DataFrame, collated_data:pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time taken(mins) for redirection per session_id
    """
    
    redirection_message_id = data.loc[~pd.isnull(data["redirected"]), ["message_n", "session_id"]]
    
    redirection_message_id = redirection_message_id.rename({"message_n":"redirected_message_n"}, axis=1)
    
    redirected_data = data.merge(redirection_message_id, on="session_id")
    
    redirected_data = redirected_data.merge(unlabelled_data[["time", "session_id", "message_n"]], on=["session_id", "message_n"])
    
    redirected_data = redirected_data.loc[redirected_data["redirected_message_n"] >=  redirected_data["message_n"], :]
    
    conv_time = redirected_data.groupby("session_id")["time"].apply(_get_time_diff).reset_index()
    
    conv_time = conv_time.rename({"time":"redirected_time"}, axis=1)
    
    return collated_data.merge(conv_time, on="session_id")
