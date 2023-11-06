from Integration.BottomUp import schema_integration, match_indotti_GMT, to_GMM, genera_LAT, confronta_source_GoldStandard
from Evaluation.Evaluation import valuta, toA_B
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, CalcoloDeiCluster
from Blocking.EntityMatching import EntityMatching
from SchemaMatching import SchemaMatching
import pandas as pd, random

# bottom-up approach


# local schemas
src_links = [
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/BottomUp_3/_S1.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/BottomUp_3/_S2.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/BottomUp_3/_S3.csv'
]

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

gold_standard=pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/BottomUp_3/GoldStandard_B.csv').astype(str)
LAT=genera_LAT(SOURCES)
confronta_source_GoldStandard(gold_standard, LAT)
gmm_gs = to_GMM(gold_standard)
print(gmm_gs)
_VisualizzaDistribuzioneCluster(CalcoloDeiCluster(toA_B(gold_standard)))

# Corrispondenza N-1, sia da S2 che da S3.


sim_combinations = []
res = []

# Visto il tipo di mapping quella che reputo essere la
# migliore generazione di corrispondenze è la TOP-K-2

score = 'SimMax'
corr_method = "TOP_K_2" # TOP_K_2 # STAB_MARR
x = 20
y =3
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


exit(1)


"""
1a sim.:

[{'value_overlap': 'SJ'}, {'label': 'LEV'}, {'label': 'JAC'}]
   MT  TP  FP  FN    P       R       F
0  20  20   0   8  1.0  0.7143  0.8333
SOURCE              S1             S2               S3
GAT                                                   
1               [name]         [nome]               []
2       [new_location]     [location]       [location]
3               [type]         [type]               []
4                   []             []  [address_phone]
5                   []         [city]           [city]
6                   []             []      [name_type]
7                   []  [descrizione]               []
8                   []    [indirizzo]               []
9                   []     [telefono]               []

I falsi negativi mancati potrebbero essere legati ad un metodo ancora non sperimentato dalla grid search.
Riprovo ad eseguire il codice, altrimenti inizio a lavorare sulle soglie.
Non cambia nulla, adesso analizzo le feature che non possono andare in similitudine:



"""

exit(1)