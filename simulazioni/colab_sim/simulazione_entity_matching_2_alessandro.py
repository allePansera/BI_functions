from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, VisualizzaCluster
import pandas as pd



path='http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/Sintetici/'
src_links = [
path+'S1_clean_.csv',
path+'S2_clean_.csv',
path+'S3_clean_.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

SOURCES['S3']=SOURCES['S3'][SOURCES['S3'].id!='S3_2']
GoldStandard=pd.read_csv(path+"GoldStandardClean.csv")

GoldStandard = GoldStandard[['l_id', 'r_id']]




# come prima cosa occorre studiare la natura dei dataset che si hanno a disposizione
# avendo dei first name/second name come prima cosa inizio a creare un attributo comune 'fullname'
for s in SOURCES:
    # print(SOURCES[s].columns)
    # creazione di una variabile mix per avere confronto con nome e cognome assieme
    SOURCES[s]['fullname'] = SOURCES[s]['given_name']+' '+SOURCES[s]['surname']
    SOURCES[s]['dateofbirth'] = SOURCES[s]['date_of_birth']

# come primo tentativo per eseguire il matching provo ad usare un rules based matcher
# blocking rules - blocco su fullname
method = "join_blocker"
# method = "rule_based_diff_type"
blocking_keys = ["fullname"]
blocking_rules = []
# blocking_rules = [[{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "fullname", "score": "0.3"}]]
# matching rules
matching_rules = [
        [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "fullname", "score": "0.3"},
         {"rule": "{}_{}_exm(ltuple, rtuple) == {}", "attr": "dateofbirth", "score": "1"},]
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
cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
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


"""
Nel primo tentativo eseguito vedo che ho molti falsi negativi (6).
Il problema è probabilmente legato ad un algoritmo in fase di sim. ext. troppo stringente.

Per fare in modo che tutto sia corretto è necessario visualizzare i FN per capire dove si stringe troppo.

Il problema è probabilmente legato alle date di nascita che non coincidono tra record uguali.
La situazione è peggiorata in quanto sono saliti anche i FP, malissimo.

Torniamo a rimettere dateofbirth, il problema però rimane capire perchè record identici non vengano rilevati.

"""
exit(1)