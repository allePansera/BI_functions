from copy import deepcopy
import warnings
import networkx as nx
import pandas as pd
import numpy as np
import string
import re
import py_stringmatching as sm
import py_entitymatching as em
import py_stringsimjoin as ssj
from tqdm import tqdm

warnings.filterwarnings('ignore')


def sim_table(TableA: pd.DataFrame, TableB: pd.DataFrame):
    A = pd.DataFrame({"A": TableA.columns})
    B = pd.DataFrame({"B": TableB.columns})
    S = A.merge(B, how='cross')
    return S


def random_sim_table(TableA: pd.DataFrame, TableB: pd.DataFrame):
    S = sim_table(TableA, TableB)
    S["sim"] = np.random.rand(len(S))
    return S


def to_sim_table(SimMatrix: pd.DataFrame):
    return SimMatrix.stack().reset_index(name="sim")


def to_sim_matrix(SimTable: pd.DataFrame):
    return SimTable.pivot(index="A", columns="B", values="sim") \
        .rename_axis(None, axis=1).rename_axis(None, axis=0)


def string_preprocess(s: str, char: str = string.punctuation, word: list = []):
    if type(s) is str:
        s = s.lower()
        for c in char:
            s = s.replace(c, " ")
        for w in word:
            s = s.replace(w, " ")
    else:
        s = str(s)
    s = re.sub(" +", " ", s)
    return s.strip()


def to_GMT(MT: pd.DataFrame):
    GMT = pd.DataFrame(columns=['GAT', 'SOURCE', 'LAT', 'SLAT'])

    for x in MT.index:
        for y in MT.columns:
            for z in MT.loc[x][y]:
                GMT.loc[len(GMT)] = [x, y, z, str(y) + '_' + z]
    return GMT


def to_GMM(GMTA: pd.DataFrame):
    df = GMTA.groupby(['GAT', 'SOURCE'])['LAT'].agg(list).unstack('SOURCE')
    for c in df.columns:
        df.loc[df[c].isnull(), [c]] = df.loc[df[c].isnull(), c].apply(lambda x: [])
    return df


# per generare la Local Attribute Table
def generaLAT(Sources):
    LAT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
    for x in Sources:
        for y in Sources[x].columns:
            LAT.loc[len(LAT)] = [str(x), str(y), str(x) + '_' + str(y)]
    return LAT


def IsKey(R, K, Numero):
    if Numero:
        return R.groupby(K).size().reset_index(name='counts').query("counts>1").shape[0]
    else:
        return R.groupby(K).size().reset_index(name='counts').query("counts>1")


def IsFD(R, A, Det, Numero):
    if Numero:
        return R.groupby(A)[Det].nunique().reset_index(name='counts').query("counts>1").shape[0]
    else:
        return R.groupby(A)[Det].nunique().reset_index(name='counts').query("counts>1")


def ConfrontaSourceGoldStandard(_GoldStandard, _LAT):
    FOJ = pd.merge(_GoldStandard, _LAT, how='outer', indicator=True).sort_values(['GAT', 'SOURCE'])

    print("I seguenti LAT non sono mappati in GAT")
    print(FOJ.query(" _merge == 'right_only' ")['SLAT'].tolist())

    print("I seguenti GAT in SOURCE sono mappati in più LAT")
    print(IsFD(_GoldStandard, ['GAT', 'SOURCE'], 'LAT', 0))

    print("I seguenti LAT in SOURCE sono mappati in più GAT")
    print(IsFD(_GoldStandard, ['LAT', 'SOURCE'], 'GAT', 0))


#############################################################
######################## LABEL-BASED ########################
#############################################################
def levenshtein_sim(row: pd.Series):
    lev = sm.Levenshtein()
    return lev.get_sim_score(
        string_preprocess(row["A"]),
        string_preprocess(row["B"])
    )


def levenshtein_label_based_similarity(TableA: pd.DataFrame, TableB: pd.DataFrame):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(levenshtein_sim, axis=1)
    return C.sort_values("sim", ascending=False)


# Jaro
def jaro_sim(row: pd.Series):
    jaro = sm.Jaro()
    return jaro.get_sim_score(
        string_preprocess(row["A"]),
        string_preprocess(row["B"])
    )


def jaro_label_based_similarity(TableA: pd.DataFrame, TableB: pd.DataFrame):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(jaro_sim, axis=1)
    return C


# Jaccard
def jaccard_sim(row: pd.Series):
    jac = sm.Jaccard()
    tok = sm.WhitespaceTokenizer(return_set=True)
    return jac.get_sim_score(
        tok.tokenize(string_preprocess(row["A"])),
        tok.tokenize(string_preprocess(row["B"]))
    )


def jaccard_label_based_similarity(TableA: pd.DataFrame, TableB: pd.DataFrame):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(jaccard_sim, axis=1)
    return C.sort_values("sim", ascending=False)


# OverlapCoefficient
def OC_sim(row: pd.Series):
    oc = sm.OverlapCoefficient()
    tok = sm.WhitespaceTokenizer(return_set=True)
    return oc.get_sim_score(
        tok.tokenize(string_preprocess(row["A"])),
        tok.tokenize(string_preprocess(row["B"]))
    )


def OC_label_based_similarity(TableA: pd.DataFrame, TableB: pd.DataFrame):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(OC_sim, axis=1)
    return C.sort_values("sim", ascending=False)


# JaroWinkler
def JaroWinkler_sim(row: pd.Series):
    jw = sm.JaroWinkler()
    return jw.get_sim_score(
        string_preprocess(row["A"]),
        string_preprocess(row["B"])
    )


def JaroWinkler_label_based_similarity(TableA: pd.DataFrame, TableB: pd.DataFrame):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(JaroWinkler_sim, axis=1)
    return C.sort_values("sim", ascending=False)


# MongeElkan
# Secondary similarity function. This is expected to be a sequence-based similarity measure
# (defaults to Jaro-Winkler similarity measure).
def MongeElkan_sim(row: pd.Series):
    me = sm.MongeElkan()
    tok = sm.WhitespaceTokenizer(return_set=True)
    return me.get_raw_score(
        tok.tokenize(string_preprocess(row["A"])),
        tok.tokenize(string_preprocess(row["B"]))
    )


def MongeElkan_label_based_similarity(TableA: pd.DataFrame, TableB: pd.DataFrame):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(MongeElkan_sim, axis=1)
    return C.sort_values("sim", ascending=False)


#############################################################
############# LOCAL SINGLE ATTRIBUTE STRATEGIES #############
#############################################################

def thresholding(SimTable: pd.DataFrame, threshold: float):
    return SimTable[SimTable["sim"] > threshold].sort_values(["sim"], ascending=[False])


def top_K(SimTable: pd.DataFrame, K: int, AoB: str = "A"):
    MT = deepcopy(SimTable)
    MT["pos"] = MT.sort_values(["sim"], ascending=[False]).groupby(AoB).cumcount()
    return MT[MT["pos"] < K].drop(columns=["pos"]).sort_values([AoB, "sim"], ascending=[True, False])


def top_1(SimTable: pd.DataFrame, AoB: str = "A"):
    return top_K(SimTable, 1, AoB)


#############################################################
###################### GLOBAL MATCHING ######################
#############################################################


def stable_marriage(MatchTable: pd.DataFrame):
    MATCH = pd.DataFrame(columns=["A", "B", "sim"])
    MT = deepcopy(MatchTable)
    MT = MT.sort_values(["sim"], ascending=[False])
    while True:
        R = MT.loc[(~MT["A"].isin(MATCH["A"])) & (~MT["B"].isin(MATCH["B"]))]
        if len(R) == 0:
            break
        x = R.iloc[0, :]
        MATCH = MATCH.append(x, ignore_index=True)
    return MATCH


def simmetric_best_match(MatchTable: pd.DataFrame):
    CMT = deepcopy(MatchTable)

    CMT['A_RowNo'] = CMT.sort_values(['sim'], ascending=[False]) \
                         .groupby(['A']) \
                         .cumcount() + 1

    CMT['B_RowNo'] = CMT.sort_values(['sim'], ascending=[False]) \
                         .groupby(['B']) \
                         .cumcount() + 1

    return CMT[(CMT.A_RowNo == 1) & (CMT.B_RowNo == 1)].drop(columns=['A_RowNo', 'B_RowNo']).sort_values(['sim'], ascending=[False])


#############################################################
########################## OVERLAP ##########################
#############################################################


def jaccard_sim_value(row: pd.Series, TableA: pd.DataFrame, TableB: pd.DataFrame):
    j = sm.Jaccard()
    return j.get_raw_score(
        TableA[row["A"]].apply(string_preprocess).tolist(),
        TableB[row["B"]].apply(string_preprocess).tolist()
    )


def jaccard_value_overlap_sim(TableA: pd.DataFrame, TableB: pd.DataFrame):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(jaccard_sim_value, args=(TableA, TableB), axis=1)
    return C.sort_values("sim", ascending=False)


def generalized_sim_value(row: pd.Series, TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    j = sm.GeneralizedJaccard(
        sim_func=sm.Levenshtein().get_sim_score,
        threshold=threshold
    )
    return j.get_raw_score(
        TableA[row["A"]].apply(string_preprocess).tolist(),
        TableB[row["B"]].apply(string_preprocess).tolist()
    )


def generalized_value_overlap_sim(TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(generalized_sim_value, args=(TableA, TableB, threshold), axis=1)
    return C.sort_values("sim", ascending=False)


def funzione_similarita_internaLEV(row: pd.Series):  # Levenshtein
    lev = sm.Levenshtein()
    return lev.get_sim_score(
        string_preprocess(row["AX"]),
        string_preprocess(row["AY"])
    )


def extended_value_overlap_sim_LEV(row: pd.Series, TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    TX = TableA[[row["A"]]].applymap(string_preprocess).drop_duplicates()
    TY = TableB[[row["B"]]].applymap(string_preprocess).drop_duplicates()
    TX.columns = ['AX']
    TY.columns = ['AY']
    PCC = TX.drop_duplicates().merge(TY.drop_duplicates(), how='cross')
    PCC["SimJac"] = PCC.apply(funzione_similarita_internaLEV, axis=1)
    INTERSEZIONE = PCC[PCC.SimJac >= threshold]
    SoloInAX = PCC.loc[~PCC['AX'].isin(INTERSEZIONE['AX'])][['AX']].drop_duplicates()
    SoloInAY = PCC.loc[~PCC['AY'].isin(INTERSEZIONE['AY'])][['AY']].drop_duplicates()
    return len(INTERSEZIONE) / (len(SoloInAX) + len(SoloInAY) + len(INTERSEZIONE))


def value_overlap_extended_jaccard_LEV(TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(extended_value_overlap_sim_LEV, args=(TableA, TableB, threshold), axis=1)
    return C.sort_values("sim", ascending=False)


def funzione_similarita_internaJaccard(row: pd.Series):  # Jaccard
    jac = sm.Jaccard()
    tok = sm.WhitespaceTokenizer(return_set=True)
    return jac.get_sim_score(
        tok.tokenize(string_preprocess(row["AX"])),
        tok.tokenize(string_preprocess(row["AY"])))


def extended_value_overlap_sim_JAC(row: pd.Series, TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    TX = TableA[[row["A"]]].applymap(string_preprocess).drop_duplicates()
    TY = TableB[[row["B"]]].applymap(string_preprocess).drop_duplicates()
    TX.columns = ['AX']
    TY.columns = ['AY']
    PCC = TX.drop_duplicates().merge(TY.drop_duplicates(), how='cross')
    PCC["SimJac"] = PCC.apply(funzione_similarita_internaJaccard, axis=1)
    INTERSEZIONE = PCC[PCC.SimJac >= threshold]
    SoloInAX = PCC.loc[~PCC['AX'].isin(INTERSEZIONE['AX'])][['AX']].drop_duplicates()
    SoloInAY = PCC.loc[~PCC['AY'].isin(INTERSEZIONE['AY'])][['AY']].drop_duplicates()
    return len(INTERSEZIONE) / (len(SoloInAX) + len(SoloInAY) + len(INTERSEZIONE))


def value_overlap_extended_jaccard_JAC(TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(extended_value_overlap_sim_JAC, args=(TableA, TableB, threshold), axis=1)
    return C.sort_values("sim", ascending=False)


def sim__join(row: pd.Series, TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    TX = TableA[[row["A"]]].applymap(string_preprocess).drop_duplicates()
    TY = TableB[[row["B"]]].applymap(string_preprocess).drop_duplicates()
    TX.columns = ['AX']
    TY.columns = ['AY']

    INTERSEZIONE = ssj.jaccard_join(TX, TY,  # tabelle su cui effettuare il sim join
                                    'AX', 'AY',  # chiavi delle tabelle
                                    'AX', 'AY',  # attributi di join
                                    sm.WhitespaceTokenizer(return_set=True),
                                    threshold=threshold,
                                    show_progress=False,
                                    l_out_attrs=['AX'], r_out_attrs=['AY']
                                    )
    SoloInAX = TX.loc[~TX['AX'].isin(INTERSEZIONE['l_AX'])][['AX']].drop_duplicates()
    SoloInAY = TY.loc[~TY['AY'].isin(INTERSEZIONE['r_AY'])][['AY']].drop_duplicates()
    return len(INTERSEZIONE) / (len(SoloInAX) + len(SoloInAY) + len(INTERSEZIONE))


def value_overlap_simjoin_jaccard(TableA: pd.DataFrame, TableB: pd.DataFrame, threshold: float):
    C = sim_table(TableA, TableB)
    C["sim"] = C.apply(sim__join, args=(TableA, TableB, threshold), axis=1)
    return C.sort_values("sim", ascending=False)


#############################################################
#################### COMBINED APPROACHES ####################
#############################################################


def max_sim_table(SimTableList: list):
    ST = pd.DataFrame(columns=["A", "B", "sim"])
    for x in SimTableList:
        ST = ST.append(x, ignore_index=True)
    return ST.groupby(["A", "B"])["sim"].max().reset_index()


def min_sim_table(SimTableList: list):
    ST = pd.DataFrame(columns=["A", "B", "sim"])
    for x in SimTableList:
        ST = ST.append(x, ignore_index=True)
    return ST.groupby(["A", "B"])["sim"].min().reset_index()


def avg_sim_table(SimTableList: list):
    ST = pd.DataFrame(columns=["A", "B", 'sim'])
    for x in SimTableList:
        ST = ST.append(x, ignore_index=True)
    return ST.groupby(["A", "B"])["sim"].mean().reset_index()


def Weighted_sum(dataframes, weights):
    if len(dataframes) < 2 or len(dataframes) != len(weights):
        raise ValueError("È necessario fornire almeno due DataFrame e i relativi pesi.")

    result_df = dataframes[0].copy()

    result_df["sim"] = result_df["sim"] * weights[0]

    for i in range(1, len(dataframes)):
        df = dataframes[i]
        weight = weights[i]

        result_df = pd.merge(result_df, df, on=["A", "B"], how="outer")

        result_df["sim"] = result_df.apply(
            lambda row: row["sim_x"] + row["sim_y"] * weight if not pd.isnull(row["sim_x"]) and not pd.isnull(row["sim_y"]) else row[
                "sim_x"] if not pd.isnull(row["sim_x"]) else row["sim_y"] * weight, axis=1)

        result_df = result_df.drop(columns=["sim_x", "sim_y"])

    return result_df


#############################################################
######################## VALUTAZIONE ########################
#############################################################

def Valuta(Gold: pd.DataFrame, Match: pd.DataFrame):
    Gold.columns = Match.columns = ['A', 'B']
    FOJ = Gold.merge(Match, how='outer', indicator=True)

    TP = FOJ[FOJ['_merge'] == 'both']
    FP = FOJ[FOJ['_merge'] == 'right_only']
    FN = FOJ[FOJ['_merge'] == 'left_only']

    if len(TP) == 0:
        return pd.DataFrame({
            'MT': [len(Match)],
            'TP': [len(TP)],
            'FP': [len(FP)],
            'FN': [len(FN)],
            'P': [round(0, 4)],
            'R': [round(0, 4)],
            'F': [round(0, 4)]
        })
    else:
        P = len(TP) / (len(TP) + len(FP))
        R = len(TP) / (len(TP) + len(FN))
        F = 2 * P * R / (P + R)
        return pd.DataFrame({
            'MT': [len(Match)],
            'TP': [len(TP)],
            'FP': [len(FP)],
            'FN': [len(FN)],
            'P': [round(P, 4)],
            'R': [round(R, 4)],
            'F': [round(F, 4)]
        })


def Vedi_Valuta(Gold: pd.DataFrame, Match: pd.DataFrame, metrics: str):
    Gold.columns = Match.columns = ['A', 'B']
    FOJ = pd.merge(Gold, Match, how='outer', indicator=True)

    TP = FOJ[FOJ['_merge'] == 'both']
    FP = FOJ[FOJ['_merge'] == 'right_only']
    FN = FOJ[FOJ['_merge'] == 'left_only']

    if metrics == 'FP':
        return FP
    if metrics == 'TP':
        return TP
    if metrics == 'FN':
        return FN


#############################################################
##################### SCHEMA INTEGRAION #####################
#############################################################


def ClusterComponentiConnessi(MatchTable, TuttiInodi):
    MatchTable.columns = ['A', 'B']

    Singleton = set(TuttiInodi) - set(MatchTable['A']).union(set(MatchTable['B']))

    # Creazione del grafo a partire dagli elementi della MatchTable
    G = nx.Graph()
    for _, row in MatchTable.iterrows():
        G.add_edge(row['A'], row['B'])
    #        G.add_edge(row['A'], row['B'], weight=row['sim'])
    # Aggiungi il peso (etichetta) basato su 'sim'

    # Aggiungi gli elementi singleton all'insieme dei nodi
    for element in Singleton:
        G.add_node(element)

        # Calcola i componenti connessi (clusters)
    clusters = list(nx.connected_components(G))

    # Creazione del DataFrame dei cluster
    cluster_data = {'ClusterKey': [], 'ClusterElement': []}
    for i, cluster in enumerate(clusters):
        for element in cluster:
            cluster_data['ClusterKey'].append(i + 1)
            cluster_data['ClusterElement'].append(element)

    cluster_df = pd.DataFrame(cluster_data)
    return cluster_df


def fromClusterToGMT(Cluster, LAT):
    GMT = pd.merge(Cluster, LAT, \
                   left_on='ClusterElement', right_on='SLAT'). \
        rename(columns={'ClusterKey': 'GAT'}). \
        drop(columns='ClusterElement')
    return GMT.sort_values(['GAT', 'SOURCE'])


#############################################################
################# VALUTAZIONE DEL CLUSTERING ################
#############################################################

def MatchIndottiGMT(GMT):
    Join = pd.merge(GMT, GMT, on='GAT')
    Join = Join[Join.SLAT_x <= Join.SLAT_y]
    Join = Join[['SLAT_x', 'SLAT_y']]
    Join.columns = ['A', 'B']

    return Join.drop_duplicates()
