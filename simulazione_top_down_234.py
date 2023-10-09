from Integration.TopDown import global_match_table
from Evaluation.Evaluation import valuta, toA_B
from SchemaMatching import SchemaMatching
import pandas as pd, random

# top-down approach

global_schema=pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/GlobalSchemaTopDown23.csv').astype(str)
# empty is necessary if i only need label comparison
empty_global_schema = global_schema

# local schemas
SOURCES={}
SOURCES['S1'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS1_B.csv").astype(str)
SOURCES['S2'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS2_B.csv").astype(str)
SOURCES['S3'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS3_B.csv").astype(str)
SOURCES['S4'] = pd.read_csv("http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/CameraS4_B.csv").astype(str)



LAT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
for x in SOURCES.keys():
    for y in SOURCES[x].columns:
      LAT.loc[len(LAT)]=[str(x),str(y), str(x)+'_'+str(y)]

sim_combinations = []
weights = [0.7, 0.15, 0.15]
corr_method = "STAB_MARR"
# score = "SimWeight"
score = 'SimMax'
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
    # print("="*50)
    # sim_methods = [{"label": "JARO_WINK"}, {"value_overlap": "GEN_JAC"}, {"value_overlap": "SJ"}]
    # print(sim_methods)

    global_match = global_match_table(SOURCES, empty_global_schema, sim_methods, corr_method, score, weights)

    # global_match_sorted = global_match.sort_values("Sim. Score", ascending=[False])
    # score evaluation
    # gold standard is global 1-1 for this test
    gold_standard = pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/EE_GoldStandardTopDown234.csv').astype(str)
    final_result = valuta(toA_B(gold_standard), toA_B(global_match))
    # print(final_result)
    res.append([final_result, sim_methods])
    # print("=" * 50)

res = sorted(res, key=lambda x: sum(x[0]["F"]), reverse=True)
for i in range(3):
    print("="*50)
    print(res[i][1])
    print(res[i][0])

