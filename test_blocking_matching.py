from Blocking.EntityMatching import EntityMatching
import pandas as pd


path='http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/NCVR2/'

src_links = [
path+'NCVR_AF_clean.csv',
path+'NCVR_BF_clean.csv',
path+'NCVR_CF_clean.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }

GoldStandardCLEAN = pd.read_csv(path + "GoldStandardCLEAN2.csv")


# print(GoldStandardCLEAN.columns)
for s in SOURCES:
    # print(SOURCES[s].columns)
    # creazione di una variabile mix per avere confronto con nome e cognome assieme
    SOURCES[s]['full_name'] = SOURCES[s]['first_name']+' '+SOURCES[s]['last_name']

# blocking rules - blocco su zip_code
method = "overlap_blocking"
blocking_keys = ["full_name"]
blocking_rules = []
# matching rules
matching_rules = [
        [{"rule": "{}_{}_lev_sim(ltuple, rtuple) >= {}", "attr": "last_name", "score": "0.7"}, {"rule": "{}_{}_exm(ltuple, rtuple) == {}", "attr": "zip_code", "score": "1"}], # first and rule
            # or
        [{"rule": "{}_{}_lev_sim(ltuple, rtuple) >= {}", "attr": "last_name", "score": "0.3"}, {"rule": "{}_{}_lev_sim(ltuple, rtuple) >= {}", "attr": "first_name", "score": "0.3"}]
] # second and rule

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
