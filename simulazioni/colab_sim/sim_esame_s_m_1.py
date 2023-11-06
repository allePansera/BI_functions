from Integration.BottomUp import schema_integration, match_indotti_GMT, to_GMM, genera_LAT, confronta_source_GoldStandard
from Evaluation.Evaluation import valuta, toA_B
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, CalcoloDeiCluster
from Blocking.EntityMatching import EntityMatching
from SchemaMatching import SchemaMatching
import pandas as pd, random

# bottom-up approach


# local schemas
src_links = [
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/BottomUp_2/S1.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/BottomUp_2/S2.csv'
]

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

gold_standard = pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/BottomUp_2/GoldStandard_A.csv').astype(str)

# come prima cosa guardo il tipo di cardinalità presente nel GS per scegliere la corrispondenza migliore
# stampa per verificare il tipo di mapping tra le sorgenti e il gold standard
LAT=genera_LAT(SOURCES)
confronta_source_GoldStandard(gold_standard, LAT)
# se il gold standard ha una corrispondenza 1-1 anche devo applicare una 1-1
# devi studiare la differenza tra stable marriage e symm best matches
gmm_gs = to_GMM(gold_standard)

_VisualizzaDistribuzioneCluster(CalcoloDeiCluster(toA_B(gold_standard)))

# Ho un mapping di tipo N-1 in quanto con S2
# ['location', 'indirizzo'] e ['nome', 'descrizione'] sono
# mappati in due attributi

sim_combinations = []
res = []

score = 'SimMax'
corr_method = "TOP_K_2"
x = 15
y = 2
# create x possible combination of y different sim. measure
for i in range(x):
    cur_list = []
    for j in range(y):
        dup = True
        # pick an item 0 or 1 to choose between label and value_overlap
        while dup:
            type = "label" if random.randint(0, 1) else "value_overlap"
            sim = SchemaMatching.SUPPORTED_METHOD[type][random.randint(0, len(SchemaMatching.SUPPORTED_METHOD[type])-1)]
            combo = {type: sim}
            if combo not in cur_list:
                cur_list.append(combo)
                dup = False
    sim_combinations.append(cur_list)


sim_combinations = [
    [{'label': 'LEV'}, {'value_overlap': 'GEN_JAC'}],
    [{'label': 'JARO_WINK'}, {'value_overlap': 'GEN_JAC'}],
    [{'label': 'JARO_WINK'}, {'value_overlap': 'SJ'}]
                    ]
for sim_methods in sim_combinations:
    global_match_table = schema_integration(SOURCES, sim_methods, corr_method, score)
    final_result = valuta(match_indotti_GMT(gold_standard), match_indotti_GMT(global_match_table))
    res.append([final_result, sim_methods, global_match_table])

res = sorted(res, key=lambda x: sum(x[0]["F"]), reverse=True)
for i in range(3):
    print("="*50)
    print(res[i][1])
    print(res[i][0])
    gmm = to_GMM(res[i][2])
    print(gmm)


"""
Devo ridurre il numero di falsi negativi. 
Per ridurre i falsi negativi è necessario ridurre le soglie con cui valuto la similarità.
N.B.: Rischio di far salire chiaramente i FP ma vediamo fino a che punto ci si può spingere.

Cur. Sol.:
[{'value_overlap': 'JAC'}, {'label': 'LEV'}, {'value_overlap': 'GEN_JAC'}]
   MT  TP  FP  FN    P       R       F
0  19  19   0   8  1.0  0.7037  0.8261
SOURCE           S1                   S2
GAT                                     
1            [addr]                   []
2            [city]               [city]
3        [location]           [location]
4            [name]  [nome, descrizione]
5       [name_type]                   []
6        [telefone]           [telefono]
7                []          [indirizzo]
8                []               [type]

Cambiando il combiner con un AVG otteniamo un risultato peggiore.
Cambiare il tipo di matching porta solo errori introducendo una corrispondenza 1:1.
Lo proviamo solo per dimostrare quanto affermato.
In realtà siamo peggiorati ma non tanto come atteso...

Ri-eseguiamo 5 volte lo script originale per verificare se esistono altre combinazioni ottime non testate.

Casi ottimi:

[{'label': 'OC'}, {'label': 'LEV'}, {'value_overlap': 'SJ'}]
   MT  TP  FP  FN       P       R       F
0  26  22   4   5  0.8462  0.8148  0.8302
SOURCE                 S1                         S2
GAT                                                 
1                  [addr]                         []
2                  [city]                     [city]
3              [location]                 [location]
4       [name_type, name]  [descrizione, type, nome]
5              [telefone]                 [telefono]
6                      []                [indirizzo]

Continuo a non trovare combinazioni che mi permettano di ottenere meno meno di FN

Provo alcune combinazioni a mano.

Noto che di solito i risultati buoni si ottengono con:
- JACCARD
- SIM JOIN
- JARO_WINK
- LEV

Niente, LEV e JAC sono il top ma non trovo combo migliori...

"""

exit(1)