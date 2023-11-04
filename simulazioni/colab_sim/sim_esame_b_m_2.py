from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, VisualizzaCluster
import pandas as pd
from copy import deepcopy

path='http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/NCVR2/'

src_links = [
path+'NCVR_AF_clean.csv',
path+'NCVR_BF_clean.csv',
path+'NCVR_CF_clean.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

GoldStandard=pd.read_csv(path + "GoldStandardCLEAN2.csv")

# come prima cosa calcolo il cluster del golden standard
cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
                                                                 EntityMatching.id_sources(SOURCES))
# visualizzo la cardinalità e la distro. del gold standard

# visualizzare distribuzione del cluster -> printa in output
_VisualizzaDistribuzioneCluster(cluster_gold_standard)

# visualizzazione dei cluster
grouped_cluster = VisualizzaCluster(cluster_gold_standard)


# E' un dataset con dei nomi e cognomi
# Creo una var. joint con full-name
for s in SOURCES:
    # print(SOURCES[s].columns)
    # creazione di una variabile mix per avere confronto con nome e cognome assieme
    SOURCES[s]['fullname'] = SOURCES[s]['first_name']+' '+SOURCES[s]['last_name']
    # per un limite della mia scarsa capacità a programmare tolgo gli underscore
    SOURCES[s]['birthplace'] = SOURCES[s]['birth_place']


# come primo tentativo per eseguire il matching provo ad usare un rules based matcher
# blocking rules - blocco su fullname
method = "join_blocker"
# method = "rule_based_diff_type"
blocking_keys = ["fullname"]
blocking_rules = []
# blocking_rules = [[{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "fullnamezip", "score": "0.3"}]]
# matching rules
# devo avere lo stesso luogo di nascita (non cambia, zip code si e se ho due persone
# registrate con sesso diverso c'è da piangere)
matching_rules = [
        [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "fullname", "score": "0.3"},
         # vorrei mettere l'exm ma non è disponibile (secondo me perchè ci sono dei valori nulli)
         {"rule": "{}_{}_lev_sim(ltuple, rtuple) >= {}", "attr": "birthplace", "score": "0.3"}]
]
# l attrs to exclude
omit_l_attrs = ["l_id"]
# r attrs to exclude
omit_r_attrs = ["r_id"]
# matching method
matching_method = "STAB" # STAB # SYMM

"""em = EntityMatching(SOURCES)
entity_match_table = em.process(method, blocking_keys, omit_l_attrs, omit_r_attrs,
                                blocking_rules, matching_rules, matching_method)

# query per verifica del matching all'interno della match table
CorrespMT = entity_match_table.groupby('r_id')[['l_id']].count().sort_values('l_id', ascending=False)


cluster_entity_matching=EntityMatching.cluster_componenti_connessi(entity_match_table[['l_id','r_id']],
                                                                   EntityMatching.id_sources(SOURCES))


# da controllare per verificare Precision e Recall
Valuta = valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}))

# da controllare quando si hanno score troppo bassi nel matching
VediValuta = vedi_valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}),
                    "FN")
# da controllare per verificare il numero di elementi per ogni cluster, può essere comodo per valutare il mapping
ClusterGrouping = cluster_entity_matching.groupby('ClusterKey').apply(EntityMatching.aggregazione_cluster).reset_index().sort_values('#Elements',
                                                                                                 ascending=False)

# visualizzare distribuzione del cluster
_VisualizzaDistribuzioneCluster(cluster_entity_matching)

# visualizzazione dei cluster
grouped_cluster = VisualizzaCluster(cluster_entity_matching)"""

"""
Il primo risultato è terribile.
Devo sia Ridurre i Falsi Positivi che i Falsi negativi.

Come prima cosa provo a vedere i FN da cosa derivano...
Sembra che molti arrivino dal matching con birth_place.

Ora però ho un problema enorme con i FP... Rimetto birthplace ma abbasso la soglia.
Non cambia molto... Qui ho il problema dei cognomi diversi...


La solzuione migliore è quella di creare due SOURCES e splittare il golden standard e fare dei match differenti.
"""
# funzione per la visualizzazione in dettaglio dei casi di Falsa Negatività o positività sulla base di vedi_valuta

"""Unione = pd.DataFrame(columns=SOURCES['S1'].columns)
for x in SOURCES.keys():
    Unione = Unione.append(SOURCES[x])
vedi_valuta_detail = pd.merge(pd.merge(VediValuta.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')"""



# Inizio la creazione del SOURCES_MAN
SOURCES_MAN = {}
for s in SOURCES:
    SOURCES_MAN[s] = deepcopy(SOURCES[s])
    # rimuovo tutti i record collegati alle donne
    SOURCES_MAN[s].drop(SOURCES_MAN[s][SOURCES_MAN[s]['sex'] == 'female'].index)

em = EntityMatching(SOURCES_MAN)
entity_match_table = em.process(method, blocking_keys, omit_l_attrs, omit_r_attrs,
                                blocking_rules, matching_rules, matching_method)

# query per verifica del matching all'interno della match table
CorrespMT = entity_match_table.groupby('r_id')[['l_id']].count().sort_values('l_id', ascending=False)


cluster_entity_matching=EntityMatching.cluster_componenti_connessi(entity_match_table[['l_id','r_id']],
                                                                   EntityMatching.id_sources(SOURCES_MAN))

cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
                                                                 EntityMatching.id_sources(SOURCES_MAN))
# da controllare per verificare Precision e Recall
Valuta = valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}))

# da controllare quando si hanno score troppo bassi nel matching
VediValuta = vedi_valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}),
                    "FN")
# da controllare per verificare il numero di elementi per ogni cluster, può essere comodo per valutare il mapping
ClusterGrouping = cluster_entity_matching.groupby('ClusterKey').apply(EntityMatching.aggregazione_cluster).reset_index().sort_values('#Elements',
                                                                                                 ascending=False)

# visualizzare distribuzione del cluster
_VisualizzaDistribuzioneCluster(cluster_entity_matching)

# visualizzazione dei cluster
grouped_cluster = VisualizzaCluster(cluster_entity_matching)

Unione = pd.DataFrame(columns=SOURCES_MAN['S1'].columns)
for x in SOURCES_MAN.keys():
    Unione = Unione.append(SOURCES_MAN[x])
vedi_valuta_detail = pd.merge(pd.merge(VediValuta.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')
exit(1)