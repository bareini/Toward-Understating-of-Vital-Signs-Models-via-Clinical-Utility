import os
from collections import OrderedDict
import matplotlib.pyplot as plt

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
        fitted = Holt(values, damped_trend=True).fit(optimized=True, smoothing_level=0.7)  # type: HoltWintersResults
        trend_dict[patient] = fitted.predict(start=0, end=len(values)-1)
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
    return {patient: np.abs(val_dict[patient] - holt_dict[patient]) for patient in val_dict.keys()}


def get_max_distance(distance_dict):
    """
    method to get the maximum distance from distance dict in order to get the weights

    :param distance_dict: a dict of the distances between holt pred to trajectory of choice
    :return: the max distance value
    """
    return max(np.hstack(values for values in distance_dict.values()))


def get_weights(distance_dict, normalized=True):
    """
    given a dict of distances, normalized them to a weight using the max distance

    :param distance_dict: a dict of the distances between holt pred to trajectory of choice
    :param distance_dict: a dict of the distances between holt pred to trajectory of choice
    :return: for each patient a weight based on the normalized distance
    """
    u_max = get_max_distance(distance_dict)
    return {k: v/u_max for k, v in distance_dict.items()} if normalized else {k: v for k, v in distance_dict.items()}


def w_trend(y_true_dict, y_pred_dict, g_true_dict, g_pred_dict, return_true=False, mode='mean_raw'):
    """
    calculate for each patient his w_trend value, as the error ( |y_t - y_t^hat|) weighted by the distance from holt
    trend.

    :param y_true_dict: dict of y_t values per patient
    :param y_pred_dict: dict of y_t^hat values per patient
    :param g_true_dict: dict of the distances between holt pred to y_t
    :param g_pred_dict: dict of the distances between holt pred to y_t_pred
    :param return_true: bool, a flag whether to calculate w_true
    :param mode: str, mode of the returned value. possible values are --'mean_raw': unnormalized score /
    mean_norm': mean score normalized /'deviaions': 75th qnt of normalized score

    :return: a dict with w_trend score per patient trajectory, one for y_t and y_t^hat
    """
    w_true = {}
    w_pred = {}
    norm = 'mean_raw' != mode
    weights_true_dict = get_weights(g_true_dict, norm)
    weights_pred_dict = get_weights(g_pred_dict, norm)
    for patient, weight_true in weights_true_dict.items():
        distance = np.abs(y_true_dict[patient] - y_pred_dict[patient])
        w_pred[patient] = np.quantile(distance * weights_pred_dict[patient], 0.95, axis=0) if mode == 'deviaions' else \
            np.mean(distance * weights_pred_dict[patient])
        if return_true:
            w_true[patient] = np.quantile(distance * weights_true_dict[patient], 0.95, axis=0) if mode == 'deviaions' \
                else np.mean(distance * weights_true_dict[patient])

    return w_true, w_pred if return_true else w_pred


def mean_metric_score(metric_dict, mode='mean'):
    """
    given a dict with scores per patient, return the mean value
    
    :param metric_dict: a metric score, per patient
    :param mode: str, mode of the returned value. possible values are --'mean'/' deviaions' (75th qnt)

    :return: 
    """
    return np.mean([v for v in metric_dict.values()]) if mode == 'mean' else np.quantile([v for v in metric_dict.values()], 0.95)


def plot_graphs(outputs, tr_dict, pred_tr_dict, label, pred_label, figs_loc=None, y_label=None, x_label=None,
                threshold=20.):
    """
    method for creating the visual summary report. Each graph has y axis limit of ± threshold (default value is 20).

    :param dict outputs:
        Dictionary with the samples for the report. for each event it has a dict of patient id and time stamps.
    :param dict tr_dict:
        Dictionary which contains for each patient (key)
        a list of the true trajectory values indexed by timestamp (value).
    :param dict pred_tr_dict:
        Dictionary which contains for each patient (key)
         a list of the predicted trajectory values indexed by timestamp (value).
    :param str label: the trajectory name to print on the legend.
    :param str pred_label: The legend label of the predicted model.
    :param figs_loc: optional. If the user wishes to save the figs, it uses this dir.
    :param str x_label: optional. X label of the entire graph.
    :param str y_label: optional. Y label of the entire graph.
    :param float threshold: the threshold around the y axis. defalut value is 20.
    :return:
    """
    plt.close()
    num_of_cols = len(outputs)
    inner_size = len(list(outputs.values())[0])
    num_of_rows = inner_size
    fig, ax = plt.subplots(nrows=num_of_rows, ncols=num_of_cols, figsize=(20, 10))
    for col, (name, values) in enumerate(outputs.items()):
        print(f"Type: {name}")
        for idx, patient in values.items():
            # plt.close()
            print(f"Sample #{idx}")
            i = patient['patient']
            locs = patient['ts']

            tr_values = tr_dict[i][locs]
            pred_values = pred_tr_dict[i][locs]

            ax[idx, col].plot(locs, tr_values, label=label, color='red', marker='P', alpha=0.7)
            ax[idx, col].plot(locs, pred_values, label=pred_label, marker='.', color='navy', alpha=0.7)
            if idx == 0 and col == num_of_cols - 1:
                ax[idx, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 18, 'weight': 'bold'})
            min_val = np.round(min(np.min(tr_values), np.min(pred_values)) - threshold, 0)
            max_val = np.round(max(np.max(tr_values), np.max(pred_values)) + threshold, 0)
            ax[idx, col].set_ylim([min_val, max_val])
            ax[idx, col].set_yticks(np.arange(min_val, max_val, 5), minor=True)
            ax[idx, col].set_yticks(np.arange(min_val, max_val, 20), minor=False)
            ax[idx, col].tick_params(axis='both', labelsize=16)
            ax[idx, col].grid(b=True, color='black', linestyle='-', lw=0.15, which='major')
            ax[idx, col].grid(b=True, color='black', alpha=0.5, linestyle='-', lw=0.1, which='minor')
    fig.tight_layout()
    if y_label is not None or x_label is not None:
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        if y_label is not None:
            plt.xlabel(x_label, fontdict={'size': 20, 'weight': 'bold'}, labelpad=20)
        if x_label is not None:
            plt.ylabel(y_label, fontdict={'size': 20, 'weight': 'bold', 'rotation': 90}, labelpad=30)
    plt.tight_layout()
    if figs_loc is not None:
        plt.savefig(os.path.join(figs_loc, f"{list(outputs.keys())}_plot.png"), dpi=400)
    plt.show()
