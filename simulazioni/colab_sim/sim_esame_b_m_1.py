from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, VisualizzaCluster
import pandas as pd



path='http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/PPRL/'
src_links = [
path+'cleanDatasetA3.csv',
path+'cleanDatasetB3.csv',
path+'cleanDatasetC3.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

GoldStandard = pd.read_csv(path+ 'cleanGoldStandard3.csv')
GoldStandard = GoldStandard.rename(columns={"id_x": "l_id", "id_y": "r_id"})
# come prima cosa calcolo il cluster del golden standard
cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
                                                                 EntityMatching.id_sources(SOURCES))
# visualizzo la cardinalità e la distro. del gold standard

# visualizzare distribuzione del cluster -> printa in output
_VisualizzaDistribuzioneCluster(cluster_gold_standard)

# visualizzazione dei cluster
grouped_cluster = VisualizzaCluster(cluster_gold_standard)

# E' un dataset con dei nomi e cognomi e anche con delle date di nascita
# Creo una var. joint con full-name ed inoltro sistemo date_of_birth rimuovendo gli underscore
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
# blocking_rules = [[{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "fullnamezip", "score": "0.3"}]]
# matching rules
matching_rules = [
        [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "fullname", "score": "0.3"},
         {"rule": "{}_{}_exm(ltuple, rtuple) == 1", "attr": "dateofbirth", "score": "0.3"}]
]
# l attrs to exclude
omit_l_attrs = ["l_id"]
# r attrs to exclude
omit_r_attrs = ["r_id"]
# matching method
matching_method = "STAB" # STAB # SYMM

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

"""
Il primo tentativo è ottimo, ho solo un falso negativo.
Il falso negativo, guardando vedi_valuta, è:

s1_10: elisabett,domiten,19081008
s2_10: elisabet,domitienn,19071008

Non ci si può fare molto essendo la data sbagliata e
nome+cognome scritti con dei typos.

Riducendo le soglie potremmo rischiare di andare a introdurre
un elevato numero di falsi positivi.

I test realizzabili potrebbero essere legati ad una riduzione della soglia di matching
oppure rimuovendo dal BoolMatcher l'exact matching della DoB.

"""
# funzione per la visualizzazione in dettaglio dei casi di Falsa Negatività o positività sulla base di vedi_valuta
# unione lo posso popolare esclusivamente impiegando le tabelle che singolarmente mi danno errore quindi staticamente.
# Il falso positivo si trova tra S2 ed S1 quindi costruisco unione su misura.
Unione = pd.DataFrame(columns=SOURCES['S2'].columns)
for x in SOURCES.keys():
        Unione = Unione.append(SOURCES[x])
vedi_valuta_detail = pd.merge(pd.merge(VediValuta.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')
exit(1)