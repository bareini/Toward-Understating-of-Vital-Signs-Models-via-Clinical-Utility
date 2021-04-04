import warnings

import pandas as pd
import numpy as np


class VSG:
    """
    This class implements the VSG algorithm, which allows to select clinical relevant samples of continuous vital sign
    trajectories.
    This class also allows the user to create random report
    """
    def __init__(self, f_norm_dict, g_norm_dict, t_l=6, t_b=4, candidates=5, n_g=4, n_f=4):
        """
        Initiate a new VSG instance with the relevant tuning parameters

        :param dict f_norm_dict:
            Dict which contains for each patient id (key), a list of f_norm values (values)
        :param dict g_norm_dict:
            Dict which contains for each patient id (key), a list of g_trend values (values)
        :param t_l:
            indicates what the total number of events to include in the selected trajectory
            (not including the selected event)
        :param t_b:
            indicates how many events should be presented prior to the selected event
        :param candidates:
            The number of candidates to consider for diversity each step
        :param n_g:
            The number of events to select from g_trend extreme values bucket
        :param n_f:
            The number of events to select from f_norm extreme values bucket
        """
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

        self._tf = {}
        self._tg = {}

    @property
    def tf(self):
        return self._return_format(self._tf)

    @property
    def tg(self):
        return self._return_format(self._tg)

    def _return_format(self, ret_dict):
        """
        transforms the events from df_a index to "ts" (index of the trajectory lists) for external use

        :param dict ret_dict: the selected events time stamp
        :return:
        """
        return {
            idx: {
                'patient': self.df_a.loc[events[0], 'patient'],
                'ts': self.df_a.loc[events, 'ts'].values}
            for idx, events in ret_dict.items()
        }

    def sorted_metrics(self, value):
        """
        sorts the values based on the requested utility component ('f' or 'g'), and returns df_a
        indexes in the corresponding order.

        :param value: the utility component name
        :return: list of indexes
        """
        return self.df_a[f'{value}_value'].abs().sort_values(ascending=False).index.tolist()

    def update_inner_diver(self, event_index):
        """
        update the diversity of f_norm values (some should be positive changes, some negative)

        :param event_index: the index of the selected event
        :return:
        """
        if self.df_a.loc[event_index, 'f_value'] > 0:
            self.tf_plus -= 1
        else:
            self.tf_minus -= 1

    def extend_trajectory(self, events):
        """
        given list of events, extend each of them based on length of trajectory, and return a list contain all events.
        this list is used to ensure there is no overlap

        :param events: list of events to extend
        :return: list of all events
        """
        extended_list = []
        for e in events:
            temp = self.extend_event(e)
            extended_list.extend(temp)
        return extended_list

    def extend_event(self, e):
        """
        given an event, extend it to the full trajectory based on events before and after parameters

        :param e: event to extend (index of df_a)
        :return: list of the trajectory
        """
        e_id, e_ts = self.df_a.loc[e, ['patient', 'ts']]
        temp = self.df_a.loc[(self.df_a['patient'] == e_id) &
                             (self.df_a['ts'].between(e_ts - self.t_before, e_ts + self.t_after))].index.tolist()
        return temp

    def take_next_candidate(self, current_events, possible_candidates, f_candidates=False, num_candidates=None):
        """
        given the current selected events, select the next C candidates (whether its from class param or given)
        that doesn't overlap with current events (neither the candidate event nor its extended trajectory).
        For candidates for the f_norm bucket, it also ensure there is enough budget for it sign.
        returns list of candidates

        :param current_events: list of the current selected events
        :param possible_candidates: list of possible candidates to consider
        :param f_candidates: whether it should consider the sign of the f value
        :param num_candidates: optional. number of candidate to consider (used in random case)
        :return: list of candidate events
        """
        extended = self.extend_trajectory(current_events)
        temp_c = self.candidates
        if num_candidates is not None:
            temp_c = num_candidates
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

    def get_diversified(self, candidates, current_events, val_name):
        """
        selects the event that diversify the current events values, using the following formula:

        :math:`argmax_{e \\in S_c} |E(S) - E(S \\bigcup {U(e)_{component}})|.`

        :param candidates: the candidates to consider
        :param current_events: the current selected events
        :param val_name: which values to diversify
        :return: the event that diversify the count
        """
        other_value = f'{val_name}_value'
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
        """
        create a dict which contains for each event, all of his extended events.

        :param events:
        :return:
        """
        return {idx: self.extend_event(event) for idx, event in enumerate(events)}

    def calc_vsg(self):
        """
        The VSG algorithm, which takes the parameters of the class and set of events
        and finds the most "relevant" and diverse events, based on their f_norm and g_trend values.

        :return:
        """
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
        self._tf = self.events_to_dict(tf)
        self._tg = self.events_to_dict(tg)

    def get_random_sample(self, n_rand):
        """
        for random trajectory, given a number of random samples to plot,
        return similar output as VSG algorithm

        :param n_rand: number of events to return
        :return:
        """
        t_rand = []
        l_rand = self._get_random_list()
        while len(t_rand) < n_rand:
            if len(t_rand) == 0:
                e = l_rand.pop(0)
            else:
                e = self.take_next_candidate(t_rand, l_rand, num_candidates=1)[0]
                l_rand.pop(l_rand.index(e))
            t_rand.append(e)
        t_dict = self.events_to_dict(t_rand)
        return self._return_format(t_dict)

    def _get_random_list(self):
        """
        randomize the order of the candidate list (df_a indexes) for creating random sample
        :return:
        """
        return self.df_a.sample(frac=1).index.tolist()

    @staticmethod
    def create_f_df(utility_component_dict, val_type):
        """
        take a dict of metric per patient and return a dataframe of values, patient id and timestamp

        :param dict utility_component_dict: metric per patient dict
        :param val_type: the clinical utility component label
        :return:
        """
        return (pd.DataFrame
                .from_dict(utility_component_dict, orient='index')
                .unstack()
                .reset_index(name=f'{val_type}_value')
                .rename(columns={'level_0': 'ts', 'level_1': 'patient'})
                .dropna()
                )

    @staticmethod
    def create_df_matrix(f_norm_dict, g_norm_dict):
        """
        create A from algorithm - a dataframe with patient id, time stamp and components values (f,g)

        :param dict f_norm_dict:
            Dict which contains for each patient id (key), a list of f_norm values (values)
        :param dict g_norm_dict:
            Dict which contains for each patient id (key), a list of g_trend values (values)
        :return:
        """
        df_f = VSG.create_f_df(f_norm_dict, val_type='f')
        # change f to reflect change in f norm value
        df_f['old_f_val'] = df_f['f_value']
        df_f_temp = df_f.sort_values(['patient', 'ts']).groupby(['patient'])['f_value'].diff().fillna(0)
        df_f.loc[df_f_temp.index, 'f_value'] = df_f_temp
        df_g = VSG.create_f_df(g_norm_dict, val_type='g')
        df_a = pd.merge(df_f, df_g, how='inner', on=['patient', 'ts'])
        return df_a
