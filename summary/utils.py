import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt, HoltWintersResults

from sklearn.utils import deprecated


def de_norm(pred, criteria, location=None):
    """
    Denormalize the time series, using saved std and mean

    :param pred: the normalized prediction
    :param criteria: metric type (e.g. sys BP)
    :param location: the folder location of the std and mean dataframes.
    :return:
    """
    if location is None:
        location = os.path.join(os.path.abspath(os.getcwd()), 'data')
    df_std = pd.read_pickle(os.path.join(location, 'df_std.pkl'))
    df_mean = pd.read_pickle(os.path.join(location, 'df_mean.pkl'))
    std = df_std[criteria]
    mean = df_mean[criteria]
    return (pred * std) + mean


@deprecated("This is old version. Don't Use!")
def trend(tr_dict):
    """
    version of holt calculation, which fit and predict for each new sample

    :param tr_dict:
    :return:
    """
    trends_dict = OrderedDict()

    for k, v in tr_dict.items():
        pred_des = [v[0], v[1]]
        for i in range(2, len(v)):
            des = Holt(v[:i]).fit(optimized=True)
            pred_des.append(des.forecast(2)[0])
        trends_dict[k] = np.array(pred_des)
    return trends_dict


def get_trend(tr_dict, return_fitted=False):
    """
    method of trend calculation, which fit holt
    which returns also g_trend (i.e. the distance between y_holt - y_trajectory)

    :param tr_dict:
    :param return_fitted:
    :return:
    """
    trend_dict = OrderedDict()
    res_dict = None
    if return_fitted:
        res_dict = OrderedDict()
    for patient, values in tr_dict.items():
        fitted = Holt(values).fit(optimized=True)  # type: HoltWintersResults
        trend_dict[patient] = fitted.predict()
        if return_fitted:
            res_dict[patient] = fitted.summary()
    return trend_dict, res_dict


# calc trend utility
@deprecated("get_g_norm is more generic")
def g_trend(tr, trend):
    return np.sqrt(np.square(tr - trend))


def f_norm_naive_sys(tr):
    """
    calculate the f norm for each trajectory value (in sys BP settings) in naive way:
    the value is:
     * y_t > 180: 1
     * 90 <= y_t <= 180: 0
     * y_t < 90: -1

    :param tr:
    :return:
    """
    tr = de_norm(tr, criteria='P_sys_q0.5')
    mask_hyper = (tr > 180)
    mask_hypo = (tr < 90) * -1
    mask = mask_hyper + mask_hypo
    return mask.astype(int)


def f_norm_per_patient(f1, f2):
    """
    per patient trajectory it calculates the f_norm score, for each side of trajectory

    :param f1: the first trajectory (predication or real value)
    :param f2: the second trajectory (predication or real value)
    :return:
    """
    temp_a = f1 * f2
    return np.mean(np.maximum(0, abs(f1) - abs(f2)) + np.where(temp_a < 0, 1, 0))


def f_norm_eval(dict1, dict2):
    """
    Return the mean value of all the patients f_norm value, between two trajectories

    We should consider making it per patient as well.

    :param dict1: first dict of trajectories (predication or real value) per patient
    :param dict2: second dict of trajectories (predication or real value) per patient
    :return: mean value per patient
    """
    sum_norm_total = 0
    for patient, values in dict1.items():
        sum_norm_total += f_norm_per_patient(values, dict2[patient])
    return sum_norm_total / len(dict1)


def get_g_norm(val_dict, holt_dict):
    """
    Calculates the g_norm for each trajectory (i.e. Y_t or Y_t^hat) as
    |y_trajectory - y_holt|, for each value and patient and return a dict per patient.
    can be changed to RMSE.

    :param val_dict: the dict of trajectories
    :param holt_dict: the dict of value of the holt prediction
    :return: a dict of distance between the trajectory to holt
    """
    return {np.abs(val_dict[patient] - holt_dict[patient]) for patient in val_dict.keys()}


def get_max_distance(distance_dict):
    """
    method to get the maximum distance from distance dict in order to get the weights

    :param distance_dict: a dict of the distances between holt pred to trajectory of choice
    :return: the max distance value
    """
    return max(np.hstack(values for values in distance_dict.values()))


def get_weights(distance_dict):
    """
    given a dict of distances, normalized them to a weight using the max distance

    :param distance_dict: a dict of the distances between holt pred to trajectory of choice
    :return: for each patient a weight based on the normalized distance
    """
    u_max = get_max_distance(distance_dict)
    return {k: v/u_max for k, v in distance_dict.items()}


def w_trend(y_true_dict, y_pred_dict, g_true_dict, g_pred_dict):
    """
    calculate for each patient his w_trend value, as the error ( |y_t - y_t^hat|) weighted by the distance from holt
    trend.

    :param y_true_dict: dict of y_t values per patient
    :param y_pred_dict: dict of y_t^hat values per patient
    :param g_true_dict: dict of the distances between holt pred to y_t
    :param g_pred_dict: dict of the distances between holt pred to y_t_pred
    :return: a dict with w_trend score per patient trajectory, one for y_t and y_t^hat
    """
    w_true = {}
    w_pred = {}
    weights_true_dict = get_weights(g_true_dict)
    weights_pred_dict = get_weights(g_pred_dict)
    for patient, weight_true in weights_true_dict.items():
        distance = np.abs(y_true_dict[patient] - y_pred_dict[patient])
        w_true[patient] = np.mean(distance * weight_true)
        w_pred[patient] = np.mean(distance * weights_pred_dict[patient])
    return w_true, w_pred


def mean_metric_score(metric_dict):
    """
    given a dict with scores per patient, return the mean value
    
    :param metric_dict: a metric score, per patient 
    :return: 
    """
    return np.mean(v for v in metric_dict.values())