from SchemaMatching import SchemaMatching
import pandas as pd

table_a = pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv')
table_b = pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/imdb.csv')
schema_match = SchemaMatching(table_a, table_b)

# schema matching - label similarities with string compare
# sim_table = schema_match.label_based_sim()
# print("="*45)
# print("Top 10 label similarities with label comparison:")
# print(sim_table.head(10))
# print("="*45)

# schema matching - values overlap
# sim_table = schema_match.value_overlap_sim()
# print("="*45)
# print("Top 10 label similarities with value overlap:")
# print(sim_table.head(10))
# print("="*45)

# schema matching - label similarities with Hybrid approach
# methods = [{"label": "LEV"}, {"label": "JARO"}, {"value_overlap": "JAC"}]
# sim_table = schema_match.hybrid_sim(methods=methods)
# print("="*45)
# print("Top 10 label similarities with hybrid approach:")
# print(sim_table.head(10))
# print("="*45)

# schema matching - Ensemble similarities
methods = [{"label": "LEV"}, {"label": "JARO"}, {"value_overlap": "JAC"}]
sim_table = schema_match.ensemble_sim(methods=methods)
print("="*45)
print("Top 10 label similarities with ensemble approach:")
print(sim_table.head(20))
print("="*45)

