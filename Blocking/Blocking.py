import pandas as pd
from .blocking_method import *

class Blocking:

    METHOD = [
        "eq_blocker",
        "join_blocker",
        "record_linkage",
        "overlap_blocking",
        "black_box_lev",
        "rule_based",
        "rule_based_diff_type"
    ]

    RULES = [
        "{}_{}_lev_sim(ltuple, rtuple) < {}",
        "{}_{}_lev_dist(ltuple, rtuple) < {}",
        "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}",
        "{}_{}_mel(ltuple, rtuple) < {}"
    ]

    def __init__(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame):
        """
        Constructor only initialize class attributes.

        Parameters
        ----------
        dataset_a: pd.DataFrame
            first dataset to compare

        dataset_b: pd.DataFrame
            second dataset to compare
        """
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.candidate_set = None

    def gen_blocking(self, method, blocking_keys=[], omit_l_attrs=[], omit_r_attrs=[], rules=[]):
        """
        Compute blocking rules for Entity matching.
        Keys and attrs must be pre-processed outside this function.
        :param method: method to be used, if not in METHOD exception is raised
        :param blocking_keys: one or more key to be used for blocking
        :param omit_l_attrs: attr to remove from attr comparison
        :param omit_r_attrs: attr to remove from attr comparison
        :param rules: optional rules for blocking ->
        [
            [{"rule": RULES_A, "score": SCORE_A, "attr": ATTR}, {...}], OR CONDITION
            [{"rule": RULES_c, "score": SCORE_c, "attr": ATTR}] AND CONDITION
        ]
        :return:
        """

        if method not in Blocking.METHOD: raise Exception(f"Method '{method}' is not defined")

        elif method == "eq_blocker":
            return equivalence_blocker(self.dataset_a, self.dataset_b,
                                       blocking_keys[0] if isinstance(blocking_keys, list) else  blocking_keys,
                                       l_attrs=[i for i in self.dataset_a.columns if i not in omit_l_attrs],
                                       r_attrs=[i for i in self.dataset_b.columns if i not in omit_r_attrs]
                                       )

        elif method == "join_blocker":
            return join_blocker(self.dataset_a, self.dataset_b, blocking_keys)

        elif method == "record_linkage":
            return record_linkage_blocking(self.dataset_a, self.dataset_b, blocking_keys)

        elif method == "overlap_blocking":
            return overlap_blocking(self.dataset_a, self.dataset_b,
                                    blocking_keys[0] if isinstance(blocking_keys, list) else blocking_keys,
                                    l_attrs=[i for i in self.dataset_a.columns if i not in omit_l_attrs],
                                    r_attrs=[i for i in self.dataset_b.columns if i not in omit_r_attrs],
                                    q_gram=3, overlap_size=2)

        elif method == "black_box_lev":
            return black_box_blocker_lev(self.dataset_a, self.dataset_b,
                                    blocking_keys[0] if isinstance(blocking_keys, list) else blocking_keys,
                                    l_attrs=[i for i in self.dataset_a.columns if i not in omit_l_attrs],
                                    r_attrs=[i for i in self.dataset_b.columns if i not in omit_r_attrs],
                                    sim_thresh=0.7)

        elif method == "rule_based":
            rules_list = []
            for rules_inner_list in rules:
                rules_list.append([rule["rule"].format(rule["attr"], rule["attr"], rule["score"]) for rule in rules_inner_list])
            return rule_based_blocker(self.dataset_a, self.dataset_b,
                               rules_list,
                               l_attrs=[i for i in self.dataset_a.columns if i not in omit_l_attrs],
                               r_attrs=[i for i in self.dataset_b.columns if i not in omit_r_attrs])

        elif method == "rule_based_diff_type":
            rules_list = []
            for rules_inner_list in rules:
                rules_list.append(
                    [rule["rule"].format(rule["attr"], rule["attr"], rule["score"]) for rule in rules_inner_list])
            return rule_based_blocker(self.dataset_a, self.dataset_b,
                                      rules_list,
                                      l_attrs=[i for i in self.dataset_a.columns if i not in omit_l_attrs],
                                      r_attrs=[i for i in self.dataset_b.columns if i not in omit_r_attrs])
