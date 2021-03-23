import os
import warnings

import pandas as pd
import numpy as np


class VSG:
    """
    calculate the vsg algorithem
    """
    def __init__(self, f_norm_dict, g_norm_dict, t_l=6, t_b=4, candidates=5, n_g=4, n_f=4):
        self.df_a = VSG.create_df_matrix(f_norm_dict, g_norm_dict)
        self.f_sorted_idx = self.sorted_metrics('f')
        self.g_sorted_idx = self.sorted_metrics('g')

        self.t_length = t_l
        self.t_before = t_b
        self.t_after = t_l - t_b
        self.candidates = candidates
        self.n_g = n_g
        self.n_f = n_f

        self.tf_plus = self.n_f / 2
        self.tf_minus = self.n_f - self.tf_plus

        self.tf = {}
        self.vg = {}
        self.tg = {}
        self.vf = {}

    def sorted_metrics(self, value):
        return self.df_a[f'{value}_value'].abs().sort_values(ascending=False).index.tolist()

    def update_inner_diver(self, event_index):
        if self.df_a.loc[event_index, 'f_value'] > 0:
            self.tf_plus -= 1
        else:
            self.tf_minus -= 1

    def extend_trajectory(self, events):
        extended_list = []
        for e in events:
            temp = self.extend_event(e)
            extended_list.extend(temp)
        return extended_list

    def extend_event(self, e):
        e_id, e_ts = self.df_a.loc[e, ['patient', 'ts']]
        temp = self.df_a.loc[(self.df_a['patient'] == e_id) &
                             (self.df_a['ts'].between(e_ts - self.t_before, e_ts + self.t_after))].index.tolist()
        return temp

    def take_next_candidate(self, current_events, possible_candidates, f_candidates=False):
        """
        to document - complex!

        :param current_events:
        :param possible_candidates:
        :param f_candidates:
        :return:
        """
        extended = self.extend_trajectory(current_events)
        temp_c = self.candidates
        candidates = []
        idx = 0
        while temp_c > 0:
            if idx >= len(possible_candidates):
                warnings.warn("End of the list and not enough candidates were found")
                assert len(candidates) > 0, "Didn't find any candidates"
                break
            candidate = possible_candidates[idx]
            candidate_extended = self.extend_event(candidate)
            if len(set(extended).intersection(set(candidate_extended))) == 0:   # no overlap
                if f_candidates:
                    sign = (self.df_a.loc[candidate, 'f_value'] > 0)
                    if (sign and self.tf_plus > 0) or (not sign and self.tf_minus > 0):
                        candidates.append(candidate)
                        temp_c -= 1
                else:
                    candidates.append(candidate)
                    temp_c -= 1
            idx += 1
        return candidates

    def get_diversified(self, candidates, current_events, other_value):
        current_mean = self.df_a.loc[current_events, other_value].mean()
        chosen = None
        best_val = 0
        for e in candidates:
            temp_res = np.abs(current_mean - self.df_a.loc[[*current_events, e], other_value].mean())
            if temp_res > best_val:
                chosen = e
                best_val = temp_res
        assert chosen is not None, "No candidate were found!"
        return chosen

    def events_to_dict(self, events):
        return {idx: self.extend_event(event) for idx, event in enumerate(events)}

    def calc_vsg(self):
        """

        :return:
        """
        # A from the algorithm

        tf = []
        vg = []
        while len(tf) < self.n_f:
            if len(tf) == 0:
                e = self.f_sorted_idx.pop(0)
            else:
                t_temp = self.take_next_candidate(current_events=tf, possible_candidates=self.f_sorted_idx,
                                                  f_candidates=True)
                e = self.get_diversified(t_temp, vg, 'g')
                self.f_sorted_idx.pop(self.f_sorted_idx.index(e))
            tf.append(e)
            vg.append(e)
            self.update_inner_diver(e)

        full_f_events = self.extend_trajectory(tf)
        self.g_sorted_idx = [event for event in self.g_sorted_idx if event not in full_f_events]
        tg = []
        vf = []
        while len(tg) < self.n_g:
            if len(tg) == 0:
                e = self.g_sorted_idx.pop(0)
            else:
                t_temp = self.take_next_candidate(current_events=tg, possible_candidates=self.g_sorted_idx,
                                                  f_candidates=False)
                e = self.get_diversified(t_temp, vf, 'f')
                self.g_sorted_idx.pop(self.g_sorted_idx.index(e))
            tg.append(e)
            vf.append(e)
        self.tf = self.events_to_dict(tf)
        self.tg = self.events_to_dict(tg)

    @staticmethod
    def create_f_df(f_norm_dict, val_type):
        """
        take a dict of metric per patient and return a dataframe of values, patient id and timestamp

        :param f_norm_dict:
        :param val_type:
        :return:
        """
        return (pd.DataFrame
                .from_dict(f_norm_dict, orient='index')
                .unstack()
                .reset_index(name=f'{val_type}_value')
                .rename(columns={'level_0': 'ts', 'level_1': 'patient'})
                .dropna()
                )

    @staticmethod
    def create_df_matrix(f_norm_dict, g_norm_dict):
        """
        create A from algorithm - a dataframe with patient id, time stamp and components values (f,g)

        :param f_norm_dict:
        :param g_norm_dict:
        :return:
        """
        df_f = VSG.create_f_df(f_norm_dict, val_type='f')
        df_g = VSG.create_f_df(g_norm_dict, val_type='g')
        df_a = pd.merge(df_f, df_g, how='internal', on=['patient', 'ts'])
        return df_a