from Integration.BottomUp import to_GMM
from Integration.BottomUp import schema_integration, match_indotti_GMT, genera_LAT, confronta_source_GoldStandard
from Evaluation.Evaluation import valuta, toA_B, vedi_valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, CalcoloDeiCluster
from SchemaMatching import SchemaMatching
import pandas as pd, random

# BOTTOM-UP

path=''
path="http://dbgroup.ing.unimore.it/EBI/Bottom1/"

src_links = [
  path + 'S1.csv',
  path + 'S2.csv',
  path + 'S3.csv'   ]

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

gold_standard=pd.read_csv(path +'GoldStandard.csv').astype(str)

gold_standard_m = to_GMM(gold_standard)
_VisualizzaDistribuzioneCluster(CalcoloDeiCluster(toA_B(gold_standard)))
LAT = genera_LAT(SOURCES)
confronta_source_GoldStandard(gold_standard, LAT)


sim_combinations = []
# weights = [0.7, 0.15, 0.15]
weights = []
corr_method = "TOP_K_2" #STAB_MARR #SYMM_MATCH
# score =
score = 'SimAvg' #SimMax #"SimMin #SimAvg
x = 10
y = 3
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

res = []

for sim_methods in sim_combinations:
    global_match_table = schema_integration(SOURCES, sim_methods, corr_method, score)
    final_result = valuta(match_indotti_GMT(gold_standard), match_indotti_GMT(global_match_table))
    res.append([final_result, sim_methods, global_match_table])

res = sorted(res, key=lambda x: sum(x[0]["F"]), reverse=True)
for i in range(3):
    if i < len(res):
        print("="*50)
        print(res[i][1])
        print(res[i][0])
        vv_FP = vedi_valuta(match_indotti_GMT(gold_standard[['GAT', 'SLAT']]), match_indotti_GMT(res[i][2][['GAT', 'SLAT']]), 'FP')
        vv_FN = vedi_valuta(match_indotti_GMT(gold_standard[['GAT', 'SLAT']]), match_indotti_GMT(res[i][2][['GAT', 'SLAT']]), 'FN')
        print(vv_FP)
        print(vv_FN)

exit(1)


"""
1 Tentativo:

[{'label': 'JAC'}, {'value_overlap': 'EXT_JAC_LEV'}, {'value_overlap': 'GEN_JAC'}]
   MT  TP  FP  FN      P    R       F
0  32  28   4   0  0.875  1.0  0.9333

- FN Assenti
- FP Registrati
28      S1_place    S2_indirizzo  right_only
29  S2_indirizzo    S2_posizione  right_only
30  S2_indirizzo  S3_phone_venue  right_only
31  S2_indirizzo     S3_position  right_only


Per migliorare provo ad applicare un combiner differente passando a Max ad Avg


- 2 tentativo

[{'label': 'JARO_WINK'}, {'label': 'JARO'}, {'value_overlap': 'EXT_JAC_JAC'}]
   MT  TP  FP  FN    P     R       F
0  21  21   0   7  1.0  0.75  0.8571

                    A                  B     _merge
1   S1_Identification     S2_descrizione  left_only
5      S2_descrizione  S2_identificativo  left_only
6      S2_descrizione     S3_appellation  left_only
14           S1_place       S2_posizione  left_only
15           S1_place        S3_position  left_only
19       S2_posizione     S3_phone_venue  left_only
21     S3_phone_venue        S3_position  left_only

La soluzione in questo caso comprende più FN legati al'introduzione di una tecnica più stringente in fase di clustering.
L'AVG è una tecnica più conservativa del Max per cui i FP diventano FN

Il GoldStandard fornito risulta essere valido in quanto lo stesso LAT non
risulta mappato su più GAT.

Il matching risulta essere 1-1 rispetto S1 - GS e 1-1 rispetto a S2 - GS;
S3 Presenta un matching di tipo N-1 in corrispondenza di position, phone_venue.

Rifaccio un ultimo tentativo alleggerendo la corrispondenza passando da SYMM a TOP 2.
La soglia di thresholding rimane 0.6

- 3 Tentativo:

[{'value_overlap': 'JAC'}, {'label': 'JARO'}, {'value_overlap': 'EXT_JAC_LEV'}]
   MT  TP  FP  FN    P       R      F
0  23  23   0   5  1.0  0.8214  0.902
Empty DataFrame
Columns: [A, B, _merge]
Index: []
                 A               B     _merge
14        S1_place    S2_posizione  left_only
15        S1_place     S3_position  left_only
16        S1_place  S3_phone_venue  left_only
19    S2_posizione  S3_phone_venue  left_only
21  S3_phone_venue     S3_position  left_only

Anche in questo caso esistono dei FN ma rispetto a prima il risultato è una via di mezzo.

Per ottenere un risultato migliore adesso si può provare ad intervenire sulle soglie.
Considerare anche il tipo di istanze presenti siccome vengono impiegate spesso nei migliori risultati delle tecniche
instance based.
"""
