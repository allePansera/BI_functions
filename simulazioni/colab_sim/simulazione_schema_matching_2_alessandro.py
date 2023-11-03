import pandas as pd
from Integration.BottomUp import schema_integration, match_indotti_GMT, to_GMM, genera_LAT, confronta_source_GoldStandard
from SchemaMatching import SchemaMatching
import pandas as pd, random
from Evaluation.Evaluation import valuta, toA_B
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, CalcoloDeiCluster
from Blocking.EntityMatching import EntityMatching
from Evaluation.Evaluation import vedi_valuta, valuta


SOURCES={}
SOURCES['S1'] = pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/E1/S1.csv').astype(str)
SOURCES['S2'] = pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/E1/S2.csv').astype(str)
SOURCES['S3'] = pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/E1/S3.csv').astype(str)

GoldStandard = pd.read_csv("https://dbgroup.ing.unimore.it/SIWS/E1/GoldStandardEsempioE1.csv").astype(str)
gs_gmm = to_GMM(GoldStandard)

# Per colpa di Phone4City3, in corrispondenza di S3, ho un matching N-N.
# E' facile vedere anche che Full Name di S3 è mappato in più GAT.
# Si veda la simulazione precedente per capire come viene definita la cardinalità delle corrispondenze.

# Per sicurezza printo anche la distribuzione ma questo ci fa già capire su quale tecnica
# di mapping conviene virare.

LAT = genera_LAT(SOURCES)
print("="*45)
print("GS V1")
confronta_source_GoldStandard(GoldStandard, LAT)
print("="*45)

# Ci sono molte cose interessati confrontando il GS con le singole LAT
# 1) Deve esserci un errore nel golden standard in quanto trattandosi di un integrazione
# bottom up non si possono avere degli attributi non presenti all'interno del GoldenStandard.
# E' sbagliato tutto ciò perchè per avere i cluster disgiunti non posso avere elementi in cluster diversi.

# Per le tecniche di clustering impiegate non potremmo fai avere una cardinalità 1-N in quanto
# i cluster devono essere disgiunti.

# Pur essendo sbagliato l'esercizio proviamo comunque a produrre un risultato accettabile.

AT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
for x in SOURCES.keys():
    for y in SOURCES[x].columns:
      LAT.loc[len(LAT)]=[str(x),str(y), str(x)+'_'+str(y)]

sim_combinations = []
score = 'SimMax'
# USO LA TOP-K (K==2)
corr_method = "TOP_K_2"
x = 10
y = 3
res = []
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
    final_result = valuta(match_indotti_GMT(GoldStandard), match_indotti_GMT(global_match_table))
    res.append([final_result, sim_methods, global_match_table])

res = sorted(res, key=lambda x: sum(x[0]["F"]), reverse=True)
for i in range(3):
    print("="*50)
    print(res[i][1])
    print(res[i][0])
    gmm = to_GMM(res[i][2])
    print(gmm)


"""
1 Tentativo malissimo:

==================================================
[{'value_overlap': 'SJ'}, {'value_overlap': 'EXT_JAC_LEV'}, {'label': 'OC'}]
   MT  TP  FP  FN       P       R       F
0  67  37  30  10  0.5522  0.7872  0.6491
SOURCE            S1               S2               S3
GAT                                                   
1       [Age2, Age1]           [Age1]           [Age2]
2           [Phone2]               []               []
3          [gender2]        [gender2]        [gender2]
4          [Sex, X2]            [Sex]               []
5               [X1]             [X1]             [X1]
6           [Phone4]               []         [Phone4]
7           [Phone3]               []         [Phone3]
8            [Name2]               []               []
9            [Name1]               []               []
10           [City1]               []          [City1]
11          [rec_id]         [rec_id]         [rec_id]
12                []               []           [Age4]
13                []          [City2]          [City2]
14                []          [City3]          [City3]
15                []  [Nome Completo]  [Nome Completo]
16                []         [Phone1]         [Phone1]
17                []           [Sex2]           [Sex2]
18                []         [gender]         [gender]
19                []           [Age3]               []
20                []             [X3]               []
==================================================
[{'value_overlap': 'GEN_JAC'}, {'value_overlap': 'EXT_JAC_LEV'}, {'label': 'OC'}]
   MT  TP  FP  FN       P       R       F
0  67  37  30  10  0.5522  0.7872  0.6491
SOURCE            S1               S2               S3
GAT                                                   
1       [Age2, Age1]           [Age1]           [Age2]
2           [Phone2]               []               []
3          [gender2]        [gender2]        [gender2]
4          [Sex, X2]            [Sex]               []
5               [X1]             [X1]             [X1]
6           [Phone4]               []         [Phone4]
7           [Phone3]               []         [Phone3]
8            [Name2]               []               []
9            [Name1]               []               []
10           [City1]               []          [City1]
11          [rec_id]         [rec_id]         [rec_id]
12                []           [Age3]               []
13                []          [City2]          [City2]
14                []          [City3]          [City3]
15                []  [Nome Completo]  [Nome Completo]
16                []         [Phone1]         [Phone1]
17                []           [Sex2]           [Sex2]
18                []             [X3]               []
19                []         [gender]         [gender]
20                []               []           [Age4]
==================================================
[{'label': 'JAC'}, {'label': 'OC'}, {'label': 'LEV'}]
    MT  TP  FP  FN       P       R       F
0  115  37  78  10  0.3217  0.7872  0.4568
SOURCE                        S1                 S2                        S3
GAT                                                                          
1                   [Age2, Age1]       [Age3, Age1]              [Age2, Age4]
2                          [Sex]        [Sex2, Sex]                    [Sex2]
3                           [X1]               [X1]                      [X1]
4                      [gender2]  [gender, gender2]         [gender, gender2]
5                       [rec_id]           [rec_id]                  [rec_id]
6       [Phone4, Phone3, Phone2]           [Phone1]  [Phone1, Phone3, Phone4]
7                        [City1]     [City3, City2]     [City1, City2, City3]
8                           [X2]                 []                        []
9                 [Name2, Name1]                 []                        []
10                            []    [Nome Completo]           [Nome Completo]
11                            []               [X3]                        []

"""

# Provo a fare un tentativo cambiando il tipo di matching, ho un numero troppo elevato di Falsi positivi
# Tento subito con un Symm. Best matches pur sapendo che saliranno i falsi negativi sperano di trovare
# un buon trade-off rispetto ai FP.


# Se la soluzione applicata non è sufficiente occorre usare la funzione vedi_valuta per capire nel dettaglio la natura
# degli errori commessi.

"""
2 Tentativo, il risultato è peggiorato e sono saliti i FP, torniamo alla TOP-K ma visualizziamo anche tutti i problemi 
riscontrati.

Si può fare poco con un Gold Standard rotto.
"""


exit(1)