"""
https://github.com/takaiyuk/notebooks/blob/master/PlotlyWrapper.ipynb
"""

import numpy as np
import plotly.offline as py
import plotly.graph_objs as go


class PlotlyWrapper:
    def __init__(self):
        self.colors = ["#3D0553", "#4D798C", "#7DC170", "#F7E642"]

    def _convert_to_str(self, arr):
        return np.array(arr, dtype=str)

    def _plotly_layout(self, title=None, xtitle=None, ytitle=None):
        return go.Layout(
            title=title,
            xaxis=dict(title=xtitle, ticklen=5, zeroline=False, gridwidth=2),
            yaxis=dict(title=ytitle, ticklen=5, gridwidth=2),
        )

    def distplot(self, data, col, bin_dict=None, title=None, xtitle=None, ytitle=None):
        trace = [
            go.Histogram(
                x=data[col].values,
                histfunc="count",
                marker=dict(color=self.colors[0]),
                xbins=bin_dict,
            )
        ]
        layout = self._plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
        fig = go.Figure(data=trace, layout=layout)
        return py.iplot(fig, show_link=False)

    def boxplot(self, data, col, title=None, xtitle=None, ytitle=None):
        trace = [go.Box(y=data[col].values, marker=dict(color=self.colors[0]))]
        layout = self._plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
        fig = go.Figure(data=trace, layout=layout)
        return py.iplot(fig, show_link=False)

    def barplot(self, data, xcol, ycol, title=None, xtitle=None, ytitle=None):
        trace = [
            go.Bar(
                x=self._convert_to_str(data[xcol].values),
                y=data[ycol].values,
                text=data[ycol].values,
                textposition="auto",
                marker=dict(
                    color=data[ycol].values,
                    colorscale="Viridis",
                    showscale=True,
                    reversescale=True,
                ),
            )
        ]
        layout = self._plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
        fig = go.Figure(data=trace, layout=layout)
        return py.iplot(fig, show_link=False)

    def countplot(self, data, col, title=None, xtitle=None, ytitle=None):
        trace = [
            go.Histogram(
                x=data[col].values, histfunc="count", marker=dict(color=self.colors[0])
            )
        ]
        layout = self._plotly_layout(title=title, xtitle=xtitle, ytitle=ytitle)
        fig = go.Figure(data=trace, layout=layout)
        return py.iplot(fig, show_link=False)

    def scatterplot(
        self, data, xcol, ycol, size=1, title=None, xtitle=None, ytitle=None
    ):
        trace = [
            go.Scatter(
                x=self._convert_to_str(data[xcol].values),
                y=data[ycol].values,
                mode="markers",
                marker=dict(
                    sizemode="diameter",
                    sizeref=1,
                    size=data[ycol].values ** size,
                    color=data[ycol].values,
                    colorscale="Viridis",
                    reversescale=True,
                    showscale=True,
                ),
                text=self._convert_to_str(data[xcol].values),
            )
        ]
        layout = go.Layout(
            autosize=True,
            title=title,
            hovermode="closest",
            xaxis=dict(title=xtitle, ticklen=5, zeroline=False, gridwidth=2),
            yaxis=dict(title=ytitle, ticklen=5, gridwidth=2),
            showlegend=False,
        )
        fig = go.Figure(data=trace, layout=layout)
        return py.iplot(fig, show_link=False)

    def lineplot(
        self,
        data,
        xcol,
        ycol,
        title=None,
        xtitle=None,
        ytitle=None,
        linewidth=2,
        rangeslider=False,
        slider_type="date",
    ):
        if rangeslider is True:
            xaxis = dict(
                title=xtitle,
                ticklen=5,
                zeroline=False,
                gridwidth=2,
                rangeslider=dict(visible=True),
                type=slider_type,
            )
        else:
            xaxis = dict(title=xtitle, ticklen=5, zeroline=False, gridwidth=2)

        if type(ycol) == list:
            trace = []
            for i in range(len(ycol)):
                t = go.Scatter(
                    x=data[xcol].values,
                    y=data[ycol[i]].values,
                    mode="lines",
                    name=data[ycol[i]].name,
                    line=dict(width=linewidth, color=self.colors[i]),
                )
                trace.append(t)
        else:
            trace = [
                go.Scatter(
                    x=data[xcol].values,
                    y=data[ycol].values,
                    mode="lines",
                    name=data[ycol].name,
                    line=dict(width=linewidth, color=self.colors[0]),
                )
            ]
        layout = go.Layout(
            title=title, xaxis=xaxis, yaxis=dict(title=ytitle, ticklen=5, gridwidth=2)
        )
        fig = go.Figure(data=trace, layout=layout)
        return py.iplot(fig, show_link=False)
