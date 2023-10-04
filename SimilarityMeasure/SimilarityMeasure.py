import pandas as pd
import numpy as np
import py_stringmatching as sm
import py_stringsimjoin as ssj
from SimilarityMeasure.PreProcessing import string_pre_process
import warnings

warnings.filterwarnings('ignore')
threshold = 0.8
lev = sm.Levenshtein()
jac = sm.Jaccard()
jaro = sm.Jaro()
jw = sm.JaroWinkler()
oc = sm.OverlapCoefficient()
me = sm.MongeElkan()
gj = sm.GeneralizedJaccard(sim_func=lev.get_sim_score, threshold=threshold)
tok = sm.WhitespaceTokenizer(return_set=True)


def levenshtein_similarity_function(x):
    """
    Lambda function used to calculate similarity between label inside simTable
    :param x: row to process
    :return: levenshtein similarity score
    """
    return lev.get_sim_score(string_pre_process(x['A']), string_pre_process(x['B']))


def jaro_similarity_function(x):
    """
    Lambda function used to calculate similarity between label inside simTable
    :param x: row to process
    :return: jaro similarity score
    """
    return jaro.get_sim_score(string_pre_process(x['A']), string_pre_process(x['B']))


def jaro_winkler_similarity_function(x):
    """
        Lambda function used to calculate similarity between label inside simTable
        :param x: row to process
        :return: jaro winkler similarity score
        """
    return jw.get_sim_score(string_pre_process(x["A"]), string_pre_process(x["B"]))


def jaccard_similarity_function(table_a: pd.DataFrame, table_b: pd.DataFrame, attr_a, attr_b):
    """
    Overlap method to use in order to consider 2 attribute similarity based on values
    :param table_a: DataFrame A
    :param table_b: DataFrame B
    :param attr_a: attr. to consider of DataFrame a
    :param attr_b: attr. to consider of DataFrame b
    :return: jaccard overlap score
    """
    return jac.get_raw_score(table_a[attr_a].values.tolist(), table_b[attr_b].values.tolist())


def jaccard_tokenize_similarity_function(x):
    """
    Label method to use in order to consider 2 attribute similarity based on token correlation
    :param x: row to process
    :return: jaccard label score
    """
    return jac.get_sim_score(tok.tokenize(string_pre_process(x["A"])), tok.tokenize(string_pre_process(x["B"])))


def overlap_coefficient_similarity_function(x):
    """
    Overlap Coefficient to detect label similarities
    :param x: row to process
    :return: overlap coefficient result
    """
    return oc.get_sim_score(tok.tokenize(string_pre_process(x["A"])), tok.tokenize(string_pre_process(x["B"])))


def monge_elkann_similarity_function(x):
    """
    Monge Elkann method to detect label similarities
    :param x: row to process
    :return: overlap coefficient result
    """
    return me.get_raw_score(tok.tokenize(string_pre_process(x["A"])), tok.tokenize(string_pre_process(x["B"])))


def generalized_jaccard_similarity_function(table_a: pd.DataFrame, table_b: pd.DataFrame, attr_a, attr_b):
    """
    Overlap method to use in order to consider 2 attribute similarity based on values
    :param table_a: DataFrame A
    :param table_b: DataFrame B
    :param attr_a: attr. to consider of DataFrame a
    :param attr_b: attr. to consider of DataFrame b
    :return: generalized jaccard overlap score
    """
    return gj.get_raw_score(table_a[attr_a].apply(string_pre_process).tolist(),
                            table_b[attr_b].apply(string_pre_process).tolist())


def similarity_join_function(table_a: pd.DataFrame, table_b: pd.DataFrame, attr_a, attr_b):
    """
    Similarity join reduces overlap analysis time
    :param table_a: DataFrame A
    :param table_b: DataFrame B
    :param attr_a: attr. to consider of DataFrame a
    :param attr_b: attr. to consider of DataFrame b
    :return: generalized jaccard overlap score
    """
    TX = table_a[[attr_a]].applymap(string_pre_process).drop_duplicates()
    TY = table_b[[attr_b]].applymap(string_pre_process).drop_duplicates()
    TX.columns = [attr_a]
    TY.columns = [attr_b]
    INTERSEZIONE = ssj.jaccard_join(TX, TY,  # tabelle su cui effettuare il sim join
                                    attr_a, attr_b,  # chiavi delle tabelle
                                    attr_a, attr_b,  # attributi di join
                                    tok,
                                    threshold=threshold,
                                    show_progress=False,
                                    l_out_attrs=[attr_a], r_out_attrs=[attr_b]
                                    )
    SoloInAX = TX.loc[~TX[attr_a].isin(INTERSEZIONE[f'l_{attr_a}'])][attr_a].drop_duplicates()
    SoloInAY = TY.loc[~TY[attr_b].isin(INTERSEZIONE[f'r_{attr_b}'])][[attr_b]].drop_duplicates()
    return len(INTERSEZIONE) / (len(SoloInAX) + len(SoloInAY) + len(INTERSEZIONE))
