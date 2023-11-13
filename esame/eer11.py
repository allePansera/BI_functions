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
            {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "Cognome", "score": "0.6"},
            {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "Nome", "score": "0.5"},
            {"rule": "{}_{}_exm(ltuple, rtuple) == {}", "attr": "CodiceBelfiore", "score": "1"}
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
La prima regola fornita dal prof è 
'Cognome_Cognome_jac_qgm_3_qgm_3(ltuple, rtuple)*0.7 + CodiceBelfiore_CodiceBelfiore_lev_sim(ltuple, rtuple)*0.3 > 0.6'

Vedo subito un problema legato ai cognoli in quando rischio di avere clienti diversi con stesso cognome (si pensi al cognome Rossi..)
Inoltre se il codice belfiore lo stimo come un codice cliente non è molto corretto dare così poco peso ad un valore che può idealmente
essere assimilato ad un codice fiscale (è un paragone con quello che posso considerare come un codice univoco stabile nel tempo)

Provo a re-inserire la regola usata nell'esercizio precedente come base di partenza.
Ci lavoro sopra e partendo dalla base di partenza precedentemente, che riporto in seguito, lavoro sulla soglia legata a nome e cognome

    MT   TP   FP  FN       P       R       F
0  401  290  111   5  0.7232  0.9831  0.8333


L'idea di lavorare con i soli codici belfiore potrebbe essere non sufficiente
in quanto ho nei FP dei dati che hanno stesso codice beliore.

Provo adesso a fare il test indicato in eer1.py in cui splitto il nome ed il cognome per l'analisi.

Siccome il problema è spesso nei cognomi analizzo quello.
Metto il cognome con soglia a 0.6 e in exm il codice belfiore; le condizioni saranno entrambe in AND.


Il risultato direi che sia molto soddisfacente rispetto all'esercizio precedente

    MT   TP  FP  FN       P       R       F
0  295  287   8   8  0.9729  0.9729  0.9729

Per dare uno sguardo rapido ai FP e FN si evince che la similarità nei cognomi rimane molto elevata e che un tentativo per ridurre
alcuni casi è considerare anche il nome.

Le date di nascita anche nei FP sono identiche, a meno che ad occhio non mi sia sfuggito qualche caso.
Ri-eseguo il caso precedente applicando il filtro in matching del nome.


Dopo cercherò una soluzione per migliorare i fn
    MT   TP  FP  FN      P       R       F
0  223  221   2  74  0.991  0.7492  0.8533

Salgono molto i FN ma non ho tempo per provare ulteriori combinazioni con le soglie.
"""
