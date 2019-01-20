import os
import numpy as np
import pandas as pd

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)  # You can plot your graphs offline inside a Jupyter Notebook Environment.
# print("plotly version:", plotly.__version__)


# ------------------------------------------------------------


## Colors of Viridis: https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html
C = ['#3D0553', '#4D798C', '#7DC170', '#F7E642']

## export plotly graphs as static images
def write_image(fig, filename, save=False, to_image=False):
    """
    ```
    # for displaying images
    img_bytes = write_image(fig, "plot.svg", to_image=True)
    from IPython.display import SVG, display  # import Image when PNG
    display(SVG(img_bytes))
    ```
    """
    import plotly.io as pio
    if save:
        if not os.path.exists("svgs")==True:
            os.mkdir("svgs")
        if not '.svg' in filename: 
            filename = filename + ".svg"
        pio.write_image(fig, 'svgs/{}'.format(filename))
    if to_image:
        if not '.svg' in filename: 
            filename = filename + ".svg"
        return pio.to_image(fig, format="svg")

## Title, X-Axis Title, Y-Axis Title
def plotly_layout(title=None, xtitle=None, ytitle=None):
    return go.Layout(title=title,
                     xaxis=dict(title=xtitle, ticklen=5, zeroline=False, gridwidth=2),
                     yaxis=dict(title=ytitle, ticklen=5, gridwidth=2))


# ------------------------------------------------------------


## Histogram
def plotly_hist(data, col, bin_dict=None, title=None, xtitle=None, ytitle=None):
    trace = [
        go.Histogram(
            x=data[col].values,
            histfunc = "count",
            marker=dict(color=C[0]),
            xbins=bin_dict,
        ),
    ]
    layout = plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
    fig = go.Figure(data=trace, layout=layout)
    return py.iplot(fig, show_link=False)


# ------------------------------------------------------------


## Boxplot
def plotly_boxplot(data, col, title=None, xtitle=None, ytitle=None):
    trace = [
        go.Box(
            y=data[col].values,
            marker=dict(color=C[0]),
        ),
    ]
    layout = plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
    fig = go.Figure(data=trace, layout=layout)
    return py.iplot(fig, show_link=False)


# ------------------------------------------------------------


## Barplot
def plotly_barplot(data, xcol, ycol, title=None, xtitle=None, ytitle=None):
    trace = [
        go.Bar(
            x=cat_to_str(data[xcol].values),
            y=data[ycol].values,
            text=data[ycol].values,
            textposition='auto',
            marker=dict(
                color=data[ycol].values,
                colorscale='Viridis',
                showscale=True,
                reversescale=True
            ),
        ),
    ]
    layout = plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
    fig = go.Figure(data=trace, layout=layout)
    return py.iplot(fig, show_link=False)


# ------------------------------------------------------------


## Countplot
def plotly_countplot(data, col, title=None, xtitle=None, ytitle=None):
    trace = [
        go.Histogram(
            x=data[col].values,
            histfunc = "count",
            marker=dict(color=C[0])
        ),
    ]
    layout = plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
    fig = go.Figure(data=trace, layout=layout)
    return py.iplot(fig, show_link=False)


# ------------------------------------------------------------


## Scatterplot
def plotly_scatterplot(data, xcol, ycol, size=1, title=None, xtitle=None, ytitle=None):
    trace = [
        go.Scatter(
            x=cat_to_str(data[xcol].values),
            y=data[ycol].values,
            mode='markers',
            marker=dict(sizemode='diameter',
                        sizeref=1,
                        size=data[ycol].values**size,
                        color=data[ycol].values,
                        colorscale='Viridis',
                        reversescale=True,
                        showscale=True
                        ),
            text=cat_to_str(data['day'].values),
        )
    ]
    layout = go.Layout(
        autosize=True,
        title=title,
        hovermode='closest',
        xaxis=dict(title=xtitle, ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title=ytitle, ticklen=5, gridwidth=2),
        showlegend=False
    )
    fig = go.Figure(data=trace, layout=layout)
    return py.iplot(fig, show_link=False)


# ------------------------------------------------------------


## Lineplot
def plotly_lineplot(data, xcol, ycol, title=None, xtitle=None, ytitle=None, rangeslider=True, slider_type='date'):
    if rangeslider==True:
        xaxis = dict(title=xtitle, ticklen=5, zeroline=False, gridwidth=2,
                     rangeslider=dict(visible=True), type=slider_type)
    else:
        xaxis = dict(title=xtitle, ticklen=5, zeroline=False, gridwidth=2)
    
    trace = [
        go.Scatter(
            x=data[xcol].values,
            y=data[ycol].values, 
            mode='lines', 
            line=dict(width=5, color=C[0])
        )
    ]
    layout = go.Layout(
        title=title,
        xaxis=xaxis,
        yaxis=dict(title=ytitle, ticklen=5, gridwidth=2))
    fig = go.Figure(data=trace, layout=layout)
    return py.iplot(fig, show_link=False)