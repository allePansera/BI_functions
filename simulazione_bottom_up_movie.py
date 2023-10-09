from Integration.BottomUp import schema_integration, match_indotti_GMT
from Evaluation.Evaluation import valuta, toA_B
from SchemaMatching import SchemaMatching
import pandas as pd, random

# bottom-up approach


# local schemas
SOURCES={}
SOURCES['S1'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS1_B.csv").astype(str)
SOURCES['S2'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS2_B.csv").astype(str)
SOURCES['S3'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS3_B.csv").astype(str)
SOURCES['S4'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS4_B.csv").astype(str)

'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS1_B.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS2_B.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS3_B.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS4_B.csv',

LAT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
for x in SOURCES.keys():
    for y in SOURCES[x].columns:
      LAT.loc[len(LAT)]=[str(x),str(y), str(x)+'_'+str(y)]

sim_combinations = []
score = 'SimMax'
corr_method = "STAB_MARR"
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

# print(sim_combinations)
# exit(1)


res = []
for sim_methods in sim_combinations:
    global_match_table = schema_integration(SOURCES, sim_methods, corr_method, score)
    gold_standard = pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/GoldStandardFull.csv').astype(str)
    final_result = valuta(match_indotti_GMT(gold_standard), match_indotti_GMT(global_match_table))
    res.append([final_result, sim_methods])

res = sorted(res, key=lambda x: sum(x[0]["F"]), reverse=True)
for i in range(3):
    print("="*50)
    print(res[i][1])
    print(res[i][0])

