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
            #{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "Nome", "score": "0.5"},
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

############ MIGLIORE RISULTATO
 -> Cognome a 0.6 con Jaccard
 -> CodiceBelfiore con exm a 1

    MT   TP  FP  FN       P       R       F
0  295  287   8   8  0.9729  0.9729  0.9729
    
Si potrebbe ragionare su una soglia inferiore quando si valutano i nomi.

Sono riportati, per il caso in cui si ha 8,8 in FN e FP (Cognome a 0.6 e CodiceBelfiore in exm) i casi d'errore

F.P.:

A_237,B_384,right_only,El-Tayeb El-Hadijj Ben,Khelfallah,04/10/1987,M,Algeria,Z301  A_237,el-tayeb el-hadijj ben khelfallah,el-tayeb el-hadijj ben khelfallah algeria,El-Tayeb El-Hadj Ben,Khrlfallah,10/04/1987,M,Algeria,Z301,B_384,el-tayeb el-hadj ben khrlfallah,el-tayeb el-hadj ben khrlfallah algeria
B_384,C_241,right_only,El-Tayeb El-Hadj Ben,Khrlfallah,10/04/1987,M,Algeria,Z301    B_384,el-tayeb el-hadj ben khrlfallah,el-tayeb el-hadj ben khrlfallah algeria,El-Tayeb El-Hadj,Khelfallah,04-10-1987,M,Algeria,Z301,C_241,el-tayeb el-hadj khelfallah,el-tayeb el-hadj khelfallah algeria
A_487,B_554,right_only,Thoiba,Singh,03/07/2004,M,India,Z222,    A_487,thoiba singh,thoiba singh india,Jagbir,Singh,02/10/1990,M,India,Z222,B_554,jagbir singh,jagbir singh india
A_583,C_564,right_only,Chika Yagazie,Chukwumefije,21/09/1986,M,Nigeria,Z335,    A_583,chika yagazie chukwumefije,chika yagazie chukwumefije nigeria,Chika,Chukwumerije,21-09-1986,M,Nigeria,Z335,C_564,chika chukwumerije,chika chukwumerije nigeria
B_443,C_486,right_only,Adriatik,Hoxha,08/23/1983,M,Albania,Z100,    B_443,adriatik hoxha,adriatik hoxha albania,Enver,Hoxha,13-05-1987,M,Albania,Z100,C_486,enver hoxha,enver hoxha albania
B_178,C_509,right_only,Fares,Al-Ferjani,04/15/1995,M,Tunisia,Z352,  B_178,fares al-ferjani,fares al-ferjani tunisia,Mohamed Ayoub,Al-Ferjani,21-03-1989,M,Tunisia,Z352,C_509,mohamed ayoub al-ferjani,mohamed ayoub al-ferjani tunisia
B_357,C_62,right_only,Salah,Al-Mejri,07/28/2001,M,Tunisia,Z352, B_357,salah al-mejri,salah al-mejri tunisia,Ahmed,Al-Mejri,21-04-2003,M,Tunisia,Z352,C_62,ahmed al-mejri,ahmed al-mejri tunisia
B_612,C_246,right_only,Filali,Mohamed,08/29/1999,M,Marocco,Z330,    B_612,filali mohamed,filali mohamed marocco,Morchid,Mohammed,22-09-1995,M,Marocco,Z330,C_246,morchid mohammed,morchid mohammed marocco

I false positive nel caso dei nord africani possono essere legati al nome, come accade per Hoxa. 
Invece per Filiai c'è il cognome riportato nel nome quindi magari è stato un typo in inserimento.
Il nome però abbiamo visto che se inserito come regola produce a livello generale un risultato drasticamente peggiore.


F.N.:
A_233,B_166,left_only,Ernesto,Sofpfici,11/05/1962,M,Italia,A944,    A_233,ernesto sofpfici,ernesto sofpfici italia,Ernesto,Soffici,05/11/1962,M,Italia,A944,B_166,ernesto soffici,ernesto soffici italia
A_72,B_225,left_only,Beppe,Morgani,26/11/1982,M,Italia,E054,    A_72,beppe morgani,beppe morgani italia,Beppe,Morgagni,11/26/1982,M,Italia,E054,B_225,beppe morgagni,beppe morgagni italia
A_162,B_406,left_only,Jean,Cipolna,28/12/1955,M,Italia,L706,    A_162,jean cipolna,jean cipolna italia,Jean,Cipolina,12/28/1955,M,Italia,L706,B_406,jean cipolina,jean cipolina italia
A_194,B_428,left_only,Luciano,Giovannetti,20/09/1965,M,Italia,F262, A_194,luciano giovannetti,luciano giovannetti italia,Marco,Giovannetti,10/09/1974,M,Italia,H501,B_428,marco giovannetti,marco giovannetti italia
A_100,C_45,left_only,Donato,Lercayri,09/11/1973,M,Italia,E472,  A_100,donato lercayri,donato lercayri italia,Donato,Lercari,09-11-1973,M,Italia,E472,C_45,donato lercari,donato lercari italia
A_495,C_265,left_only,Angelo,Bottaroy,25/12/1977,M,Italia,Z131, A_495,angelo bottaroy,angelo bottaroy italia,Angelo,Bottaro,25-12-1977,M,Italia,Z131,C_265,angelo bottaro,angelo bottaro italia
B_344,C_316,left_only,Rzvan,Petcu,05/06/1973,M,Romania,Z129,    B_344,rzvan petcu,rzvan petcu romania,Rzban,Petcu,06-05-1973,M,Romania,Z129,C_316,rzban petcu,rzban petcu romania
B_366,C_613,left_only,Egidio,Jacsuzzi,05/04/1975,M,Italia,D379, B_366,egidio jacsuzzi,egidio jacsuzzi italia,Egidio,Jacuzzi,04-05-1975,M,Italia,D379,C_613,egidio jacuzzi,egidio jacuzzi italia

I False Negative sono legati al fatto che sulla dx ci sono nome e cognome uniti all'interno di nome quindi si creano dei finti match

"""
