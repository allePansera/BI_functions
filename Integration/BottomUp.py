import pandas as pd
from tqdm import tqdm
from SchemaMatching import SchemaMatching
from CorrispBuilder.CorrisBuilder import CorrisBuilder
from threading import Thread
import networkx as nx


local_matching_table = None


def local_match_table_thread(sources, x, y, sim_methods, corr_method, score):
    global local_matching_table
    sm = SchemaMatching(sources[x], sources[y])
    LocalMatchingTableXY = sm.ensemble_sim(methods=sim_methods)
    LocalMatchingTableXY = LocalMatchingTableXY.rename(columns={score: 'Sim. Score'})
    LocalMatchingTableXY = LocalMatchingTableXY[["A", "B", "Sim. Score"]]

    cb = CorrisBuilder(LocalMatchingTableXY, top_k=1)
    if corr_method == "STAB_MARR":
        LocalMatchingTableXY = cb.stable_marriage_method()
    elif corr_method == "SYMM_MATCH":
        LocalMatchingTableXY = cb.symmetric_best_match_method()
    elif corr_method == "TOP_1":
        LocalMatchingTableXY = cb.top_k_method()
    else:
        raise Exception(f"Corresp. method '{corr_method}' not supported")

    LocalMatchingTableXY = LocalMatchingTableXY[["A", "B", "Sim. Score"]]
    LocalMatchingTableXY.columns = ['LAT_A', 'LAT_B', 'Sim. Score']
    LocalMatchingTableXY = CorrisBuilder.thresholding(LocalMatchingTableXY, 0.6)
    # si aggiungono i 2 nomi delle Local Sources matchate
    LocalMatchingTableXY['SOURCE_A'] = x
    LocalMatchingTableXY['SOURCE_B'] = y
    # e anche i SLAT
    LocalMatchingTableXY['SLAT_A'] = LocalMatchingTableXY['SOURCE_A'] + '_' + LocalMatchingTableXY['LAT_A']
    LocalMatchingTableXY['SLAT_B'] = LocalMatchingTableXY['SOURCE_B'] + '_' + LocalMatchingTableXY['LAT_B']

    # per poi aggiungerlo alla LocalMatchingTable complessiva
    local_matching_table = local_matching_table.append(LocalMatchingTableXY, sort=True)


def local_match_table(sources: list, sim_methods, corr_method, score="SimAvg"):
    """

    :param sources: list of sources to compare
    :param sim_methods: methods to build ensemble similarity method
    :param corr_method: method to create correspondence
    :param score: SimMin/SimMax/SimAvg
    :return: match table global with each source
    """
    global local_matching_table
    local_matching_table = pd.DataFrame(columns=['SOURCE_A', 'LAT_A', 'SOURCE_B', 'LAT_B', 'SLAT_A', 'SLAT_B', 'sim'])
    thread_pool = []

    for x in tqdm(sources.keys()):
        for y in tqdm(sources.keys()):
            if (x <= y):  #  x <= y per calcolare anche i matching tra una sorgente e se stessa
                t = Thread(target=local_match_table_thread,
                           args=(sources, x, y, sim_methods, corr_method, score))
                thread_pool.append(t)

    # run all thread
    for t in tqdm(thread_pool):
        t.start()

    # wait for all thread end
    for t in tqdm(thread_pool):
        t.join()

    return local_matching_table


def clustering_componenti_connessi(match_table, nodi):
    match_table.columns = ['A', 'B']

    Singleton = set(nodi) - set(match_table['A']).union(set(match_table['B']))

    # Creazione del grafo a partire dagli elementi della MatchTable
    G = nx.Graph()
    for _, row in match_table.iterrows():
        G.add_edge(row['A'], row['B'])
        # G.add_edge(row['A'], row['B'], weight=row['sim'])
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


def genera_lat(sources):
    LAT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
    for x in sources:
        for y in sources[x].columns:
            LAT.loc[len(LAT)]=[str(x),str(y), str(x)+'_'+str(y)]
    return LAT


def from_cluster_to_GMT(cluster, LAT):
    GMT = pd.merge(cluster, LAT, left_on='ClusterElement', right_on='SLAT').rename(columns={'ClusterKey': 'GAT'}).drop(columns='ClusterElement')
    return GMT.sort_values(['GAT', 'SOURCE'])


def to_GMM(GMTA: pd.DataFrame):
    df = GMTA.groupby(['GAT','SOURCE'])['LAT'].agg(list).unstack('SOURCE')
    for c in df.columns:
        df.loc[df[c].isnull(), [c]] = df.loc[df[c].isnull(), c].apply(lambda x: [])
    return df


def schema_integration(sources, sim_methods, corr_method, score="SimAvg"):
    match_table = local_match_table(sources, sim_methods, corr_method, score)[['SLAT_A', 'SLAT_B']]
    nodi = [col for df in sources.values() for col in df.columns]
    cluster = clustering_componenti_connessi(match_table, nodi)
    LAT = genera_lat(sources)

    return from_cluster_to_GMT(cluster, LAT)


def match_indotti_GMT(GMT):
    Join=pd.merge(GMT,GMT, on='GAT')
    Join=Join[Join.SLAT_x<=Join.SLAT_y]
    Join=Join[['SLAT_x','SLAT_y']]
    Join.columns=['A','B']

    return Join.drop_duplicates()
