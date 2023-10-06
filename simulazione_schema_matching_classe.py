from Integration.TopDown import global_match_table
from Evaluation.Evaluation import valuta, vedi_valuta, toA_B
import pandas as pd

# top-down approach

# global schema, it's instanced so we have to remove data
global_schema=pd.read_csv("https://dbgroup.ing.unimore.it/SIWS/E1/GlobalSchema.csv").astype(str)
empty_global_schema = global_schema[0:0]

# local schemas
SOURCES={}
SOURCES['S1']=pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/E1/S1.csv').astype(str)
SOURCES['S2']=pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/E1/S2.csv').astype(str)
SOURCES['S3']=pd.read_csv('https://dbgroup.ing.unimore.it/SIWS/E1/S3.csv').astype(str)

LAT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
for x in SOURCES.keys():
    for y in SOURCES[x].columns:
      LAT.loc[len(LAT)]=[str(x),str(y), str(x)+'_'+str(y)]

sim_methods = [{"label": "JARO_WINK"}, {"value_overlap": "GEN_JAC"}, {"value_overlap": "SJ"}]
corr_method = "TOP_1"
score = "SimMax"

global_match = global_match_table(SOURCES, empty_global_schema, sim_methods, corr_method, score)

# global_match_sorted = global_match.sort_values("Sim. Score", ascending=[False])
# score evaluation
gold_standard = pd.read_csv("https://dbgroup.ing.unimore.it/SIWS/E1/GoldStandardEsempioE1.csv").astype(str)
final_result = valuta(toA_B(gold_standard), toA_B(global_match))
print(final_result)



