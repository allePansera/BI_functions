from .BlockingMatching import BlockingMatching
from Evaluation.Evaluation import valuta_blocking
from CorrispBuilder.CorrisBuilder import CorrisBuilder
from copy import deepcopy
import pandas as pd
import networkx as nx


class EntityMatching:

    def __init__(self, sources):
        self.sources = sources

    def process(self, method, blocking_keys=[], omit_l_attrs=[], omit_r_attrs=[], rules_blocking=[], rules_matching=[], matching_method="SYMM"):
        """
        Compute blocking rules for Entity matching.
        Keys and attrs must be pre-processed outside this function.
        :param method: method to be used, if not in METHOD exception is raised
        :param blocking_keys: one or more key to be used for blocking
        :param omit_l_attrs: attr to remove from attr comparison
        :param omit_r_attrs: attr to remove from attr comparison
        :param rules_blocking: optional rules for blocking ->
        [
            [{"rule": RULES_A, "score": SCORE_A, "attr": ATTR}, {...}], OR CONDITION
            [{"rule": RULES_c, "score": SCORE_c, "attr": ATTR}] AND CONDITION
        ]
        :param rules_matching: optional rules for blocking ->
        [
            [{"rule": RULES_A, "score": SCORE_A, "attr": ATTR}, {...}], AND CONDITION
            [{"rule": RULES_c, "score": SCORE_c, "attr": ATTR}] OR CONDITION
        ]
        :param matching_method: ["SYMM", "STAB"]
        :return:
        """

        MatchTable = pd.DataFrame(columns=['l_id', 'r_id', 'Sim. Score'])

        for x in self.sources.keys():
            for y in self.sources.keys():
                if (x < y):  # x<=y nel caso dirty
                    blck_mtch = BlockingMatching(self.sources[x].rename(columns={"id": "l_id"}), self.sources[y].rename(columns={"id": "r_id"}))
                    candidate_set = blck_mtch.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs, rules_blocking)
                    match_table = blck_mtch.gen_matching(candidate_set, rules_matching)
                    match_table_formatted = match_table.query("l_id!=r_id").rename(columns={"l_id": "A", "r_id": "B"})
                    # global mapping
                    cb = CorrisBuilder(match_table_formatted)
                    if matching_method == "SYMM":
                        MTxy = cb.symmetric_best_match_method().rename(columns={"A": "l_id", "B": "r_id"})
                    elif matching_method == "STAB":
                        MTxy = cb.stable_marriage_method().rename(columns={"A": "l_id", "B": "r_id"})
                    elif "TOP_K" in matching_method:
                        k = int(matching_method.split("_")[-1])
                        cb.top_k = k
                        cb.threshold = 0.8
                        MTxy = cb.top_k_method().rename(columns={"A": "l_id", "B": "r_id"})
                        MTxy = CorrisBuilder.thresholding(MTxy, 0.7)

                    else:
                        # raise Exception(f"Matching method '{matching_method}' not supported")
                        MTxy =  match_table_formatted.rename(columns={"A": "l_id", "B": "r_id"})

                    MatchTable = MatchTable.append(MTxy[['l_id', 'r_id', 'Sim. Score']], sort=True)

        return MatchTable


    @staticmethod
    def id_sources(sources: list):
        ListaID = []
        for s in sources.keys():
            ListaID += sources[s]['id'].to_list()
        return ListaID

    @staticmethod
    def cluster_componenti_connessi(match_table, tutti_i_nodi):
        match_table = deepcopy(match_table)
        match_table.columns = ['A', 'B']
        Singleton = set(tutti_i_nodi) - set(match_table['A']).union(set(match_table['B']))
        G = nx.Graph()
        for _, row in match_table.iterrows():
            G.add_edge(row['A'], row['B'])
        #        G.add_edge(row['A'], row['B'], weight=row['sim'])  # Aggiungi il peso (etichetta) basato su 'sim'

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

    @staticmethod
    def calcola_match_indotti_cluster(Cluster):
          Join=pd.merge(Cluster,Cluster, on='ClusterKey')
          Join=Join[Join.ClusterElement_x<Join.ClusterElement_y]
          Join=Join[['ClusterElement_x','ClusterElement_y']]
          Join.columns=['l_id','r_id']

          return Join.drop_duplicates()

    @staticmethod
    def aggregazione_cluster(x):
        x['source'] = x['ClusterElement'].astype(str).str[:1]
        x['ClusterElement'] = x['ClusterElement'].astype(str)

        Campi = {
            '#Sources': x['source'].nunique(),
            'Sources': x['source'].drop_duplicates().str.cat(sep=','),
            '#Elements': x['ClusterElement'].nunique(),
            'Elements': x['ClusterElement'].str.cat(sep=',')
        }
        return pd.Series(Campi)


    @staticmethod
    def find_missed_matches(gold_standard: pd.DataFrame, C: pd.DataFrame, A: pd.DataFrame, B: pd.DataFrame, on=None,
                            left_on=None,
                            right_on=None, metric: str = 'FN') -> pd.DataFrame:
        """
        Finds missed matches from given gold standard.

        Returns
        -------
        missed_match_table: pd.DataFrame
            Table containing missed matches information.
        """
        if metric == 'FP':
            merge = 'right_only'
            rows = right_on if right_on is not None else on
        elif metric == 'FN':
            merge = 'left_only'
            rows = left_on if left_on is not None else on

        missed_match_table = \
            pd.merge(gold_standard, C, how='outer', indicator=True, on=on, left_on=left_on, right_on=right_on).query(
                f"_merge=='{merge}'")[rows]

        return pd.merge(pd.merge(missed_match_table, A, left_on=rows[0], right_on='l_id'), B, left_on=rows[1],
                        right_on='r_id')
