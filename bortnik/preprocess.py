import numpy as np
import pandas pd

def quant_smooth_df(cols, n, qs):
    """
    Функция которую необходимо передавать dtf.apply, чтобы сгенерировать новые фичи
    :params:
    cols -- (list) -- Список колонок в pd.DataFrame, для которых надо сгенерировать новые фичи
    n    -- (int)  -- Сколько необходмо брать соседних значений
    qs   -- (list) -- Список значений квантилей. зачения от 0 до 100
    :returns:
    (function)
    :usage:
    df, df.groupby('well').apply(quant_smooth_df(['DT', 'CALI', 'RHOB', 'SP_f'], 1000, (2, 50, 98)))
    """
    def f(well):
        cols_ = cols
        if type(cols_) is not list:
            cols_ = [cols_]
        x = well[cols_].values
        r = quants_smooth(x, n, qs)
        names = np.array(['']*(len(cols_)*len(qs)), dtype='S100')
        for i, q in enumerate(qs):
            names[i * len(cols_) : (i+1)*len(cols_)] = np.array([col + '_q' + str(q) for col in cols_])
        return pd.DataFrame(data=r.reshape(r.shape[0], -1), index=well.index, columns=names)
    return f


def quants_smooth(x, n, qs):
    """
    Функция для вычисления квантилей.

    :params:
    x  -- (numpy array) -- матрица фичей
    n  -- (int)         -- сколько брать соседних значений. Берутся по глубине
                           Note: для того, чтобы рассматривать также крайние значения,
                           делается падинг
    qs -- (list)        -- список значений квантилей. зачения от 0 до 100

    :returns: -- (numpy array) -- Матрица новых фичей
    """
    x_pad = np.pad(x.reshape(x.shape[0], -1), pad_width=((n,n), (0,0)), mode='edge')
    res = np.zeros((x.shape[0], x_pad.shape[1] * len(qs)))
    for i in range(res.shape[0]):
        for j, q in enumerate(qs):
            res[i, j * x_pad.shape[1] : (j+1) * x_pad.shape[1]] = np.percentile(x_pad[i : i + 2*n+1], q, axis=0)
    return res
