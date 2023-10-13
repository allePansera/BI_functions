from Blocking.Blocking import Blocking
from Evaluation.Evaluation import valuta_blocking
import pandas as pd


# symmetrical schema download
A = pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/fodors.csv')
B = pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/zagats.csv')
# ed il relativo gold standard (predicted matches)
GoldStandard = pd.read_csv('http://dbgroup.ing.unimore.it/SIWS/DataIntegration/matches_fodors_zagats.csv')

# Exploration is mandatory in order to know which index drop
# print(A.columns)
# print(B.columns)
l_id_key = "l_id"
r_id_key = "r_id"
GoldStandard.columns = ['l_id', 'r_id']
A = A.rename(columns={'id': 'l_id'})
B = B.rename(columns={'id': 'r_id'})
# define a variable for result comparison
# simulate each blocking technique
methods = Blocking.METHOD

method = None
blocking_keys = ["city"]
omit_l_attrs = [l_id_key]
omit_r_attrs = [r_id_key]
blocking_set = []


for method in methods:
    print(f"Blocking for: {method}...")
    if method == "eq_blocker":
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs)
        blocking_set.append({method: block_compare})

    elif method == "join_blocker":
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs)
        blocking_set.append({method: block_compare})

    elif method == "record_linkage":
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys)
        blocking_set.append({method: block_compare})

    elif method == "overlap_blocking":
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs)
        blocking_set.append({method: block_compare})

    elif method == "black_box_lev":
        continue
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs)
        blocking_set.append({method: block_compare})

    elif method == "rule_based":
        # define only 2 rules to compare with or condition
        rules = [[{"rule": "{}_{}_lev_sim(ltuple, rtuple) < {}", "attr": "city", "score": "0.7"},
                  {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "city", "score": "0.7"}]
                 ]
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs, rules)
        blocking_set.append({method: block_compare})

    elif method == "rule_based_diff_type":
        # define only 2 rules to compare with or condition
        rules = [[{"rule": "{}_{}_lev_sim(ltuple, rtuple) < {}", "attr": "city", "score": "0.7"},
                  {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "city", "score": "0.7"}]
                 ]
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs, rules)
        blocking_set.append({method: block_compare})

# test all techniques with all possible key then get the best score comparing with GoldStandard
result_set = []
for blocking_dict in blocking_set:
    for key in blocking_dict:
        print(f"Evaluating {key}...")
        result_set.append({"method": key, "score": valuta_blocking(A, B, blocking_dict[key], GoldStandard)})

res = sorted(result_set, key=lambda x: (sum(x["score"]["PCompletness"]), sum(x["score"]["PQuality"])), reverse=True)
for i in range(3):
    print("="*50)
    print(res[i]["method"])
    print(res[i]["score"])


