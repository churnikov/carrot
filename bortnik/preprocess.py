import numpy as np
import pandas pd
from scipy import fftpack
from sklearn.base import BaseEstimator, TransformerMixin

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

def fourier_filter(n):
    """
    Функция, которую необходимо передавать dtf.apply
    Создает новый столбец SP_f используя преобразование фурье над признаком SP
    Идея в том, что SP с ростом глубины начинает смещаться вниз. При этом происходит только смещение.
    Для того, чтобы избавиться от этого тренда, можно применить преобразование преобразование фурье, а затем обратное.
    В результате, значения сместятся к "более менее" общему среднему.

    :TODO: -- имеет смысл попробовать вычитать линейную регрессию от этого

    :params:
    n -- (int) -- Сколько значений обнулить после преобразования фурье. Таким образом избавимся от "медленных трендов".

    :returns: -- (function)

    :usage:
    dtf['SP_f'] = dtf.groupby('well').apply(fourier_filter(5))
    """
    def f(well):
        sp = well.SP
        f_sp = fftpack.fft(sp.values)
        f_sp[ : n] = 0
        f_sp[-n : ] = 0
        sp_f = fftpack.ifft(f_sp).real
        return pd.Series(data=sp_f, index=well.index, name='SP_f').to_frame()
    return f


class WindowGenerator(BaseEstimator, TransformerMixin):
    """
    Создает "окно", т.е. новые фичи
    В результате учитываются предыдущие и будущие значения наблюдений

    skearn compatible
    """
    def __init__(self, window_size, **kwargs):
        """
        :window_size: -- (int) -- сколько брать значений спереди и сзади
        """
        self.window_size = window_size

    def __process(self, x):
        """
        Фактически функция генерации фичей
        """
        n = self.window_size
        n_feat = x.shape[1]
        x_pad = np.pad(x, mode='edge', pad_width=((n,n), (0,0)))
        res = np.zeros((x.shape[0], n_feat * 2 * n))

        for i in range(x.shape[0]):
                res[i, :n_feat * n] = x_pad[i : i+n, :].flatten()
                res[i, n_feat * n:] = x_pad[i+n+1 : i+2*n+1, :].flatten()

        return res

    def fit(self, X, y=0):
        self.encoder = None
        return self

    def transform(self, X, y=0):
        return self.__process(X)
