from Blocking.EntityMatching import EntityMatching
import pandas as pd


# DATASET CLEAN
path='http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/PPRL/'

src_links = [
path+'cleanDatasetA3.csv',
path+'cleanDatasetB3.csv',
path+'cleanDatasetC3.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

GoldStandard=pd.read_csv(path+ 'cleanGoldStandard3.csv')

GoldStandard.columns=['l_id','r_id']




# print(GoldStandard.columns)
for s in SOURCES:
    # print(SOURCES[s].columns)
    # creazione di una variabile mix per avere confronto con nome e cognome assieme
    SOURCES[s]['full_name'] = SOURCES[s]['given_name']+' '+SOURCES[s]['surname']

# blocking rules - blocco su zip_code
method = "overlap_blocking"
blocking_keys = ["full_name"]
blocking_rules = []
# matching rules
matching_rules = [
        [{"rule": "{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) >= {}", "attr": "surname", "score": "0.3"}]
]
# l attrs to exclude
omit_l_attrs = ["l_id"]
# r attrs to exclude
omit_r_attrs = ["l_id"]
# matching method
matching_method = "SYMM" # STAB

em = EntityMatching(SOURCES)
entity_match_table = em.process(method, blocking_keys, omit_l_attrs, omit_r_attrs,
                                blocking_rules, matching_rules, matching_method)
print(entity_match_table)
