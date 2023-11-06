from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, VisualizzaCluster
import pandas as pd
from copy import deepcopy

path='http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/PPRL/'

src_links = [
path+'DATASET_small_dirty_A.csv',
path+'DATASET_small_dirty_B.csv',
path+'DATASET_small_dirty_C.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

# rinominiamo id in entity
for x in SOURCES.keys():
    SOURCES[x].columns=['id', 'given_name', 'surname', 'date_of_birth', 'entity']

# mettiamo tutto assieme
UNIONE=pd.DataFrame(columns=SOURCES['S1'].columns)
for x in SOURCES.keys():
          UNIONE=UNIONE.append(SOURCES[x])

# Calcoliamo il gold standard
GoldStandard=UNIONE.merge(UNIONE, on='entity')[['id_x','id_y']].query('id_x <= id_y')

GoldStandard.columns=['l_id','r_id']

# come prima cosa calcolo il cluster del golden standard
cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
                                                                 EntityMatching.id_sources(SOURCES))
# visualizzo la cardinalità e la distro. del gold standard

# visualizzare distribuzione del cluster -> printa in output
_VisualizzaDistribuzioneCluster(cluster_gold_standard)

# visualizzazione dei cluster
grouped_cluster = VisualizzaCluster(cluster_gold_standard)



"""
Come prima cosa possiamo studiare la natura dei dataset
che abbiamo supposto essere identici.

Colonne:
- id
- given_name
- surname
- date_of_birth
- entity -> è collegato all'id, ce ne freghiamo

Come prima cosa rimuoviamo gli underscore; poi dopo creiamo fullname.

"""

# da qua iniziano le mie considerazioni relative al dataset da analizzare
# Creo una var. joint con full-name
for s in SOURCES:
    # print(SOURCES[s].columns)
    # creazione di una variabile mix per avere confronto con nome e cognome assieme
    SOURCES[s]['fullname'] = SOURCES[s]['given_name']+' '+SOURCES[s]['surname']
    # per un limite della mia scarsa capacità a programmare tolgo gli underscore
    SOURCES[s]['dateofbirth'] = SOURCES[s]['date_of_birth']


# come primo tentativo per eseguire il matching provo ad usare un rules based matcher
# blocking rules - blocco su fullname
method = "join_blocker"
# method = "rule_based_diff_type"
blocking_keys = ["fullname"]
# blocking_keys = []
# blocking_rules = []
blocking_rules = [
    [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "fullname",
      "score": "0.2"}]]
# matching rules
matching_rules = [
        [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "fullname", "score": "0.2"},
         # vorrei mettere l'exm ma non è disponibile (secondo me perchè ci sono dei valori nulli)
         {"rule": "{}_{}_lev_sim(ltuple, rtuple) >= {}", "attr": "dateofbirth", "score": "0.2"}]
]
# l attrs to exclude
omit_l_attrs = ["l_id"]
# r attrs to exclude
omit_r_attrs = ["r_id"]
# matching method
matching_method = "SYMM" # STAB # SYMM

em = EntityMatching(SOURCES)
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
grouped_cluster = VisualizzaCluster(cluster_entity_matching)


Unione = pd.DataFrame(columns=SOURCES['S1'].columns)
for x in SOURCES.keys():
    Unione = Unione.append(SOURCES[x])
vedi_valuta_detail = pd.merge(pd.merge(VediValuta.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')

exit(1)



