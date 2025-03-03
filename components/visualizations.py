import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import streamlit as st

def create_histograms(df, x_column, y_column, bins=30):
    """
    Create histograms for two columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    x_column : str
        Name of the first column to plot
    y_column : str
        Name of the second column to plot
    bins : int
        Number of bins for the histograms
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the histograms
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Create histograms
    sns.histplot(df[x_column], bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram of {x_column}')
    
    sns.histplot(df[y_column], bins=bins, kde=True, ax=axes[1])
    axes[1].set_title(f'Histogram of {y_column}')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_boxplot(df, x_column, y_column, figsize=(7, 5)):
    """
    Create a boxplot
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    x_column : str
        Name of the column for x-axis
    y_column : str
        Name of the column for y-axis
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the boxplot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique categories
    categories = df[y_column].unique()
    
    # Create a colormap with distinct colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    
    # Create the boxplot with custom colors
    sns.boxplot(x=x_column, y=y_column, data=df, ax=ax, palette=colors)
    
    plt.tight_layout()
    
    return fig

def create_scatterplot(df, x_column, y_column, hue_column=None, size_column=None, figsize=(9, 9)):
    """
    Create a scatterplot
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    x_column : str
        Name of the column for x-axis
    y_column : str
        Name of the column for y-axis
    hue_column : str or None
        Name of the column for color encoding
    size_column : str or None
        Name of the column for size encoding
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the scatterplot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the scatter plot
    sns.scatterplot(
        x=x_column, 
        y=y_column, 
        data=df, 
        hue=hue_column, 
        size=size_column,
        ax=ax
    )
    
    # Place legend outside the figure
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    return fig

def create_pairplot(df, hue_column, height=2):
    """
    Create a pairplot
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    hue_column : str
        Name of the column for color encoding
    height : int
        Height of each subplot
        
    Returns:
    --------
    seaborn.axisgrid.PairGrid
        The pairplot
    """
    pairplot_fig = sns.pairplot(df, hue=hue_column, height=height)
    return pairplot_fig