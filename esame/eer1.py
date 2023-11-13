from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, VisualizzaCluster
import pandas as pd
from copy import deepcopy

path='http://dbgroup.ing.unimore.it/EBI/Cluster/'


src_links = [
path+'A.csv',
path+'B.csv',
path+'C.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

cluster_gold_standard=pd.read_csv(path+ 'ClusterGoldStandard.csv')

#GoldStandard.columns=['l_id','r_id']


# come prima cosa calcolo il cluster del golden standard
#cluster_gold_standard=EntityMatching.cluster_componenti_connessi(GoldStandard[['l_id','r_id']],
#                                                               EntityMatching.id_sources(SOURCES))
# visualizzo la cardinalità e la distro. del gold standard

# visualizzare distribuzione del cluster -> printa in output
#_VisualizzaDistribuzioneCluster(GoldStandard)

# visualizzazione dei cluster - Non applicabile siccome ho un cluster e non devo costruirmelo più
grouped_cluster = VisualizzaCluster(cluster_gold_standard)

"""
La prima operazione da eseguire è la creazione di un attributo mix che non contenga tutto bensì un set di variabili limitato.
Inserisco nome e cognome mentre per il secondo caso vado ad inserire nome, cognome e nazionalità.

"""
# Creo una var. joint con bike_name e color
for s in SOURCES:
    # print(SOURCES[s].columns)
    # creo un attributo che unisca due attributi per eseguire il blocking in sim join
    SOURCES[s]['mix'] = (SOURCES[s]['Nome']+' '+SOURCES[s]['Cognome']).str.lower()
    SOURCES[s]['mix2'] = (SOURCES[s]['Nome']+' '+SOURCES[s]['Cognome']+' '+SOURCES[s]['Nazionalita']).str.lower()



# come primo tentativo per eseguire il matching provo ad usare un rules based matcher
# blocking rules - blocco su fullname
method = "join_blocker"
# method = "rule_based_diff_type"
blocking_keys = ["mix"]
# blocking_keys = []
blocking_rules = []
# blocking_rules = [
#    [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "fullname",
#      "score": "0.2"}]]
# matching rules
matching_rules = [
        [
            {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "mix", "score": "0.5"},

            {"rule": "{}_{}_exm(ltuple, rtuple) == 1", "attr": "CodiceBelfiore", "score": "1"}
            # potrei mettere degli exm
        ]
]
# l attrs to exclude
omit_l_attrs = ["l_id"]
# r attrs to exclude
omit_r_attrs = ["r_id"]
# matching method
matching_method = "SYMM" # STAB # SYMM # TOP_K_2

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
VediValuta_FN = vedi_valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}),
                    "FN")

VediValuta_FP = vedi_valuta(EntityMatching.calcola_match_indotti_cluster(cluster_gold_standard).rename(columns={"l_id": "A", "r_id": "B"}),
                     EntityMatching.calcola_match_indotti_cluster(cluster_entity_matching).rename(columns={"l_id": "A", "r_id": "B"}),
                    "FP")
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
vedi_valuta_detail_FN = pd.merge(pd.merge(VediValuta_FN.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')

vedi_valuta_detail_FP = pd.merge(pd.merge(VediValuta_FP.rename(columns={"A": "l_id", "B": "r_id"}),
                                       Unione,
                                       left_on='l_id', right_on='id'),
                              Unione, left_on='r_id', right_on='id')

print(Valuta)
exit(1)


# vedi_valuta_detail_FP[['mix_x', 'mix_y']]
# vedi_valuta_detail_FN[['mix_x', 'mix_y']]


"""
La prima regola fornita è
city_city_lev_sim(ltuple, rtuple)*0.5 + addr_addr_lev_sim(ltuple, rtuple)*0.5 > 0.9
Non penso sia molto efficace in quanto si basa sul confronto di dati altamente variabili.

Proceso come primo tentativo a controllare mix e nazionalità che in teoria non cambiano.
A livello di regole di matching come primo tentativo procedo impiegando:

[
        [
            {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "mix", "score": "0.3"},

            {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "Nazionalita", "score": "0.2"}
            # potrei mettere degli exm
        ]
]
Sono due regole in AND.

Con metodo delle corrispondenze Symmetric Best Match 1-1

Mi aspetto di contenere il numero di FN imposto dal metodo delle corrispond. scelto tenendo basse le soglie nelle regole.

    MT   TP   FP  FN       P       R       F
0  858  217  641  78  0.2529  0.7356  0.3764

Il problema è legato all'alto numero di FP
Per ridurre i falsi positivi occorrono delle regole più stringenti per cui passo da una soglia di 0.3 per
mix ad una soglia di 0.5

Cluster con max numero di elementi: [48, 81, 94]
    MT   TP   FP  FN       P       R       F
0  429  288  141   7  0.6713  0.9763  0.7956

Il risultato è decisamente migliore ma posso fare ancora meglio, controllo su Mix e Nazionalita quale 
può essere il problema con la funzione per visualizzare i FP

Ho ancora problemi legati alla dissimilarità dei cognomi per cui mi conviene alzare la soglia.
Se non risolto analizzo nome e cognome singolaramente con soglie differenti.
Provo con mix a 0.7 se non risolvo metto in and nome e cognome con entrambi a 0.7.

    MT   TP  FP  FN       P       R       F
0  286  232  54  63  0.8112  0.7864  0.7986

Sono saliti molto i Falsi negativi ma l'F score generale è salito.

Analizzando i dati vedo che i FN sono legati a problemi di inserimento.
E.s.:
al-mehdi ben slimane,mohamed ben slimane
nikolay ivanovich paslar,nikolay paslar
mauor crenna,mauro crenna

Quindi il problema è trovare una giusta soglia in mix dopo aver stabilito se si preferiscono 
i FN o i FP.
Si potrebbero fare dei tentativi splittando nome e cognome ma non ho tempo.


Vedo che c'è un codice cliente (ipotizzo) che è CodiceBelfiore. Riduco la soglia per il nome+cognome e inserisco in and la regola,
vediamo se il risultato migliora. Essendo un codice cliente mi aspetto che corrisponda.

    MT   TP   FP  FN       P       R       F
0  401  290  111   5  0.7232  0.9831  0.8333

Il risultato è il migliore ottenuto. Non ho tempo per analizzare i FP e FN. Si può fare in fase di orale

"""
