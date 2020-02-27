import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def percent_in_rows(df):
    sums = df.apply(sum, axis=1)
    df2 = df.copy()
    for i in range(len(df)):
        df2.iloc[i, :] = df.iloc[i, :] / sums.iloc[i] * 100
    return df2


def cumplot_help(df, x, y, category, percent):
    df1 = df.loc[:, [x, y, category]]
    df2 = df1.pivot(index=x, columns=category, values=y).fillna(0)

    if percent:
        df2 = percent_in_rows(df2)
    cum1 = np.cumsum(df2, axis=1)
    zeros = pd.Series(np.zeros(len(cum1)), index=cum1.index)
    cum2 = pd.concat([zeros, cum1], axis=1).iloc[:, :-1]
    cum2.columns = df2.columns
    return df2, cum2


def do_plot(x, y, offset, horizontal=False, ax=None):
    obj = plt if ax is None else ax
    if horizontal:
        return obj.barh(x, y, left=offset)
    else:
        return obj.bar(x, y, bottom=offset)


def cumplot(df, x, y, category, horizontal=False, percent=False, ax=None, **kwargs):
    df2, cum2 = cumplot_help(df, x, y, category, percent)
    ps = [None] * len(df2.columns)
    for i in range(len(df2.columns)):
        ps[i] = do_plot(df2.index, df2.iloc[:, i],
                        cum2.iloc[:, i], horizontal, ax)
    obj = plt if ax is None else ax
    obj.legend(map(lambda p: p[0], ps), df2.columns, **kwargs)


def abline(a, b, axes):
    x_vals = np.array(axes.get_xlim())
    y_vals = a * x_vals + b
    axes.plot(x_vals, y_vals, '-')
