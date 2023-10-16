from Blocking.BlockingMatching import Blocking
from Evaluation.Evaluation import valuta_blocking
import py_entitymatching as em
import pandas as pd


A=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Movies3/csv/table_a.csv')
B=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Movies3/csv/table_b.csv')
# ed il relativo gold standard (predicted matches)
GoldStandard=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Movies3/csv/predicted_matches.csv')
#
CandidateSetDato=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/wisc_em_benchmark/839_spring19/Movies3/csv/candidate_pairs.csv')

# Exploration is mandatory in order to know which index drop

l_id_key = "_id"
r_id_key = "_id"
GoldStandard = GoldStandard.rename(columns={'id1': 'l_id', 'id2': 'r_id'})
GoldStandard = GoldStandard[['l_id', 'r_id']]
A = A.rename(columns={'_id': 'l_id'})
B = B.rename(columns={'_id': 'r_id'})
# define a variable for result comparison
# simulate each blocking technique
methods = Blocking.METHOD

method = None
blocking_keys = ["name"]
omit_l_attrs = ['l_id']
omit_r_attrs = ['r_id']
blocking_set = []

# diff name type alignement
atypesA = em.get_attr_types(A)
atypesB = em.get_attr_types(B)



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
        # diff type
        continue
        rules = [[
                  {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "name", "score": "0.7"}]
                 ]
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs, rules)
        blocking_set.append({method: block_compare})

    elif method == "rule_based_diff_type":
        # define only 2 rules to compare with or condition
        rules = [[
                  {"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}", "attr": "name", "score": "0.3"}]
                 ]
        blocking_instance = Blocking(A, B)
        block_compare = blocking_instance.gen_blocking(method, blocking_keys, omit_l_attrs, omit_r_attrs, rules)
        blocking_set.append({method: block_compare})

# test all techniques with all possible key then get the best score comparing with GoldStandard
result_set = []
for blocking_dict in blocking_set:
    for key in blocking_dict:
        print(f"Evaluating {key}...")
        if len(blocking_dict[key]) > 0 :
            result_set.append({"method": key, "score": valuta_blocking(A, B, blocking_dict[key], GoldStandard)})

res = sorted(result_set, key=lambda x: (sum(x["score"]["PCompletness"]), sum(x["score"]["PQuality"])), reverse=True)
for i in range(3):
    if i < len(res):
        print("="*50)
        print(res[i]["method"])
        print(res[i]["score"])


