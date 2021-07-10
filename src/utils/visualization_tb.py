# data manipulation
from cloudpickle.cloudpickle import instance
import numpy as np
import pandas as pd 
from collections import Counter

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# route files
import os
import sys


def plot_one_song(data_notes, data_piece, number):
    """
    Plot all different notes and rests contained in one song
    """

    clean_column = data_notes[number]
    for i in list(range(len(clean_column))):
        for elem in clean_column:
            if elem.count(".") >= 1 or elem == "REST" or elem.isdigit() == True:
                clean_column.remove(elem)

    # notes are organised by occurence
    d = Counter(clean_column)
    data = {"Note":list(d.keys()), "count": list(d.values())}
    df = pd.DataFrame(data, index=range(len(data["Note"])))
    df = df.sort_values("count", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    
    piece_title= data_piece[number]

    chart = sns.barplot(df["Note"], df["count"] ,color = "green")

    for p in chart.patches:
        chart.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
            textcoords='offset points')

    ax.set_title(f"Note distribution per piece - Piece title: {piece_title}")
    ax.set_xlabel("Notes")
    ax.set_ylabel("Ocurrence")
    plt.xticks(rotation = 90)

    plt.show()


def plot_all_songs(notes):
    """
    Plot all different notes contained in all pieces
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # copy of note_list to not delete info for the model
    clean_list = notes[::]
    contador = 0
    while contador <= len(clean_list):
        contador +=1
        for elem in clean_list:
            # remove chords and rests
            if elem.count(".") >= 1 or elem == "REST" or elem.isdigit() == True:
                clean_list.remove(elem)

    # notes are organised by occurence
    d = Counter(clean_list)
    data = {"Note":list(d.keys()), "count": list(d.values())}
    x = pd.DataFrame(data, index=range(len(data["Note"])))
    x = x.sort_values("count", ascending=False)

    chart = sns.histplot(clean_list, color = "green")

    for p in chart.patches:
        chart.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
            textcoords='offset points')

    ax.set_title("Note distribution in all pieces")
    ax.set_xlabel("Notes")
    ax.set_ylabel("Ocurrence")
    plt.xticks(rotation = 90)

    plt.show()