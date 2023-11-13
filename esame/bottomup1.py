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
corr_method = "SYMM_MATCH" #STAB_MARR #SYMM_MATCH
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

[{'value_overlap': 'GEN_JAC'}, {'value_overlap': 'EXT_JAC_LEV'}, {'value_overlap': 'EXT_JAC_JAC'}]
   MT  TP  FP  FN      P    R       F
0  32  28   4   0  0.875  1.0  0.9333
               A               B      _merge
28      S1_place    S2_indirizzo  right_only
29  S2_indirizzo    S2_posizione  right_only
30  S2_indirizzo  S3_phone_venue  right_only
31  S2_indirizzo     S3_position  right_only

Ho 4 FP legati ad una regola molto stringente per la generazione delle corrispondenze
Andando ad analizzare il gold standard però si può osservare che c'è una buona similarità
rispetto all'esercizio precedente.

Il mapping è 1-1 con S1 mentre con S2 ed S3 rimane N-1 per, rispettivamente, descrizione, identificativo 
e position, phone_venue

Quello che possiamo provare a fare, come fatto in precedenza, è provare a cambiare il tipo di combiner passando da
SimMax a Sim Avg.


"""

