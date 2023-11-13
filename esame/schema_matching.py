from Integration.BottomUp import to_GMM
from Integration.BottomUp import schema_integration, match_indotti_GMT, genera_LAT, confronta_source_GoldStandard
from Evaluation.Evaluation import valuta, toA_B, vedi_valuta
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, CalcoloDeiCluster
from SchemaMatching import SchemaMatching
import pandas as pd, random

# BOTTOM-UP

src_links = [
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS1_B.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS2_B.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS3_B.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS4_B.csv',
]

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

gold_standard=pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/GoldStandardFull.csv').astype(str)
gold_standard_m = to_GMM(gold_standard)
_VisualizzaDistribuzioneCluster(CalcoloDeiCluster(toA_B(gold_standard)))
LAT = genera_LAT(SOURCES)
confronta_source_GoldStandard(gold_standard, LAT)


sim_combinations = []
# weights = [0.7, 0.15, 0.15]
weights = []
corr_method = "TOP_K_3" #STAB_MARR #SYMM_MATCH
# score =
score = 'SimMax' #SimMax #"SimMin #SimAvg
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

