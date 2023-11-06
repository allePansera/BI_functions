from Integration.TopDown import global_match_table
from Integration.BottomUp import to_GMM
from Evaluation.Evaluation import valuta, toA_B
from SchemaMatching import SchemaMatching
import pandas as pd, random


src_links = [
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/BookA1.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/BookB1.csv',
'http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/BookC1.csv'
]

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

global_schema=pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/GlobalSchema.csv').astype(str)
gold_standard=pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/EsempioBook/GMT_GSesempio.csv').astype(str)
gold_standard_m = to_GMM(gold_standard)

LAT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
for x in SOURCES.keys():
    for y in SOURCES[x].columns:
      LAT.loc[len(LAT)]=[str(x),str(y), str(x)+'_'+str(y)]

sim_combinations = []
weights = [0.7, 0.15, 0.15]
corr_method = "TOP_K_2"
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
    global_match = global_match_table(SOURCES, global_schema, sim_methods, corr_method, score, weights)
    final_result = valuta(toA_B(gold_standard), toA_B(global_match))
    res.append([final_result, sim_methods])

res = sorted(res, key=lambda x: sum(x[0]["F"]), reverse=True)
for i in range(3):
    print("="*50)
    print(res[i][1])
    print(res[i][0])


exit(1)


"""
Al primo tentativo ho avuto una quantità assurda di falsi negativi.
Studio il GS per capire che tipo di mapping conviene adottare.
"""