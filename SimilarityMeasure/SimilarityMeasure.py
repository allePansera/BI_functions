import pandas as pd
import numpy as np
import py_stringmatching as sm
import py_stringsimjoin as ssj
from SimilarityMeasure.PreProcessing import string_pre_process
import warnings

warnings.filterwarnings('ignore')
threshold = 0.3
lev = sm.Levenshtein()
jac = sm.Jaccard()
jaro = sm.Jaro()
jw = sm.JaroWinkler()
oc = sm.OverlapCoefficient()
me = sm.MongeElkan()
gj = sm.GeneralizedJaccard(sim_func=lev.get_sim_score, threshold=threshold)
tok = sm.WhitespaceTokenizer(return_set=True)


def levenshtein_similarity_function(x, col_a="A", col_b="B"):
    """
    Lambda function used to calculate similarity between label inside simTable
    :param x: row to process
    :return: levenshtein similarity score
    """
    a_cols = x.columns.to_list() if hasattr(x, "columns") else x.index.to_list()
    b_cols = x.columns.to_list() if hasattr(x, "columns") else x.index.to_list()
    if col_a not in a_cols: col_a += '_x'
    if col_b not in b_cols: col_b += '_y'
    return lev.get_sim_score(string_pre_process(x[col_a]), string_pre_process(x[col_b]))


def jaro_similarity_function(x, col_a="A", col_b="B"):
    """
    Lambda function used to calculate similarity between label inside simTable
    :param x: row to process
    :return: jaro similarity score
    """
    if col_a not in x.index: col_a += '_x'
    if col_b not in x.index: col_b += '_y'
    return jaro.get_sim_score(string_pre_process(x[col_a]), string_pre_process(x[col_b]))


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


def jaccard_tokenize_similarity_function(x, col_a="A", col_b="B"):
    """
    Label method to use in order to consider 2 attribute similarity based on token correlation
    :param x: row to process
    :return: jaccard label score
    """
    if col_a not in x.index: col_a += '_x'
    if col_b not in x.index: col_b += '_y'
    return jac.get_sim_score(tok.tokenize(string_pre_process(x[col_a])), tok.tokenize(string_pre_process(x[col_b])))


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

def ext_jaccard_lev_sim_function(table_a: pd.DataFrame, table_b: pd.DataFrame, attr_a, attr_b):
    """
    Extendend jaccard with internal sim. function levenshtein
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
    PCC = TX.drop_duplicates().merge(TY.drop_duplicates(), how='cross')
    PCC["Sim. Score"] = PCC.apply(levenshtein_similarity_function, args=(attr_a, attr_b), axis=1)
    INTERSEZIONE = PCC[PCC["Sim. Score"] >= threshold]
    if attr_a not in INTERSEZIONE.columns: attr_a += '_x'
    if attr_b not in INTERSEZIONE.columns: attr_b += '_y'
    SoloInAX = PCC.loc[~PCC[attr_a].isin(INTERSEZIONE[attr_a])][[attr_a]].drop_duplicates()
    SoloInAY = PCC.loc[~PCC[attr_b].isin(INTERSEZIONE[attr_b])][[attr_b]].drop_duplicates()
    return len(INTERSEZIONE) / (len(SoloInAX) + len(SoloInAY) + len(INTERSEZIONE))

def ext_jaccard_jac_sim_function(table_a: pd.DataFrame, table_b: pd.DataFrame, attr_a, attr_b):
    """
    Extendend jaccard with internal jaccard function levenshtein
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
    PCC = TX.drop_duplicates().merge(TY.drop_duplicates(), how='cross')
    PCC["Sim. Score"] = PCC.apply(jaccard_tokenize_similarity_function, args=(attr_a, attr_b), axis=1)
    INTERSEZIONE = PCC[PCC["Sim. Score"] >= threshold]
    if attr_a not in INTERSEZIONE.columns: attr_a += '_x'
    if attr_b not in INTERSEZIONE.columns: attr_b += '_y'
    SoloInAX = PCC.loc[~PCC[attr_a].isin(INTERSEZIONE[attr_a])][[attr_a]].drop_duplicates()
    SoloInAY = PCC.loc[~PCC[attr_b].isin(INTERSEZIONE[attr_b])][[attr_b]].drop_duplicates()
    return len(INTERSEZIONE) / (len(SoloInAX) + len(SoloInAY) + len(INTERSEZIONE))

