import pandas as pd
from Integration.BottomUp import schema_integration, match_indotti_GMT, to_GMM, genera_LAT, confronta_source_GoldStandard
from SchemaMatching import SchemaMatching
import pandas as pd, random
from Evaluation.Evaluation import valuta, toA_B
from ESERCIZI.utils.funzioni_prof import _VisualizzaDistribuzioneCluster, CalcoloDeiCluster

path="http://dbgroup.ing.unimore.it/SIWS/DataIntegration/Esempi/TopDCamera/"


src_links = [
path+'Camera3.csv',
path+'Camera4.csv']

SOURCES = { 'S'+str(i+1) : pd.read_csv(link).astype(str) for i, link in enumerate(src_links) }
GoldStandard_v1 = pd.read_csv( path +'GS_2022_EsercizioEsameSchemaMatchingBottomUp2.csv').astype(str)
GoldStandard_v2 = pd.read_csv( path +'GS2.csv').astype(str)


# Si precisa subito che quando si parla di cardinalità N-1 si intendono N LAT in 1 GAT

# Visualizzo in prima fase il gold standard per capire il tipo di corrispondenza.
# Mi serve per capire come fare il mapping
gmm_v1 = to_GMM(GoldStandard_v1)
gmm_v2 = to_GMM(GoldStandard_v2)
# Per colpa dell'attributo S2_product abbiamo una corrispondenza N-1
# siccome brand e product name finiscono in un unico attributo globale

# Eseguo una verifica extra di sicurezza sulle corrispondenze per verificare
# se quello che ho scritto è vero

LAT=genera_LAT(SOURCES)
print("="*45)
print("GS V1")
print("="*45)
# Tutto confermato, inoltre per problemi di visibilità delle colonne
# mi era sfuggito anche autofocus su S1 con cardinalità N-1.
# Meglio vederlo ora che dopo....



# Eseguo ora il matching con il mio algoritmo basato su Grid Search per la ricerca di un risultato '''ottimo'''.

LAT = pd.DataFrame(columns=['SOURCE', 'LAT', 'SLAT'])
for x in SOURCES.keys():
    for y in SOURCES[x].columns:
      LAT.loc[len(LAT)]=[str(x),str(y), str(x)+'_'+str(y)]

sim_combinations = []
score = 'SimMax'
corr_method = "TOP_K_2"
x = 10
y = 3
res = []
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


for sim_methods in sim_combinations:
    global_match_table = schema_integration(SOURCES, sim_methods, corr_method, score)
    final_result = valuta(match_indotti_GMT(GoldStandard_v1), match_indotti_GMT(global_match_table))
    res.append([final_result, sim_methods, global_match_table])

res = sorted(res, key=lambda x: sum(x[0]["F"]), reverse=True)
for i in range(3):
    print("="*50)
    print(res[i][1])
    print(res[i][0])
    gmm = to_GMM(res[i][2])
    print(gmm)

# Per le migliori soluzioni sono printate le GMM quindi posso andare a verificare il tipo di corrispondenza.
"""
Tentativo n.1 -> (migliore di quello ottenuto dal prof.)


[{'label': 'JARO'}, {'label': 'LEV'}, {'value_overlap': 'EXT_JAC_LEV'}]
   MT  TP  FP  FN    P    R       F
0  16  16   0   4  1.0  0.8  0.8889
SOURCE              S1                S2
GAT                                     
1         [auto focus]       [autofocus]
2              [focus]                []
3              [brand]           [brand]
4       [battery type]         [battery]
5       [image format]    [image format]
6                   []  [af illuminator]
7                   []  [exposure modes]
8                   []    [product name]
==================================================
[{'value_overlap': 'EXT_JAC_LEV'}, {'value_overlap': 'SJ'}, {'label': 'JARO_WINK'}]
   MT  TP  FP  FN    P    R       F
0  16  16   0   4  1.0  0.8  0.8889
SOURCE              S1                S2
GAT                                     
1         [auto focus]       [autofocus]
2              [focus]                []
3              [brand]           [brand]
4       [battery type]         [battery]
5       [image format]    [image format]
6                   []  [af illuminator]
7                   []  [exposure modes]
8                   []    [product name]
==================================================
[{'value_overlap': 'EXT_JAC_LEV'}, {'label': 'JARO'}, {'value_overlap': 'SJ'}]
   MT  TP  FP  FN    P    R       F
0  16  16   0   4  1.0  0.8  0.8889
SOURCE              S1                S2
GAT                                     
1         [auto focus]       [autofocus]
2              [focus]                []
3              [brand]           [brand]
4       [battery type]         [battery]
5       [image format]    [image format]
6                   []  [af illuminator]
7                   []  [exposure modes]
8                   []    [product name]
"""

# Attention please, le soluzioni trovate riportano tutte lo stesso score in quanto il combiner prende in consideraz.
# quello che è il risultato massimo.

# Come possiamo notare subito l'errore è legato anche al tipo di matching che andiamo a eseguire in quanto non supporta
# corrispondenze diverse dall'1:1 per cui occorre cambiare strategia sotto questo pt. di vista.

# Nel secondo tentativo proviamo a portare la corrispondenza con il Symmetric Best Match.
# Se non dovesse essere sufficiente potremmo procedere con una variazione legata al tipo di combiner.


"""
Tentativo n.2 -> (migliore di quello ottenuto dal prof. ma non di quello ottenuto in precedenza)

==================================================
[{'label': 'ME'}, {'value_overlap': 'JAC'}, {'value_overlap': 'SJ'}]
   MT  TP  FP  FN    P    R       F
0  16  16   0   4  1.0  0.8  0.8889
SOURCE              S1                S2
GAT                                     
1         [auto focus]       [autofocus]
2       [battery type]         [battery]
3              [brand]           [brand]
4       [image format]    [image format]
5                   []  [af illuminator]
6                   []  [exposure modes]
7              [focus]                []
8                   []    [product name]
==================================================
[{'value_overlap': 'GEN_JAC'}, {'value_overlap': 'EXT_JAC_LEV'}, {'label': 'LEV'}]
   MT  TP  FP  FN    P    R       F
0  16  16   0   4  1.0  0.8  0.8889
SOURCE              S1                S2
GAT                                     
1         [auto focus]       [autofocus]
2              [focus]                []
3       [battery type]         [battery]
4              [brand]           [brand]
5       [image format]    [image format]
6                   []  [af illuminator]
7                   []  [exposure modes]
8                   []    [product name]
==================================================
[{'label': 'ME'}, {'label': 'JARO_WINK'}, {'value_overlap': 'GEN_JAC'}]
   MT  TP  FP  FN    P    R       F
0  16  16   0   4  1.0  0.8  0.8889
SOURCE              S1                S2
GAT                                     
1         [auto focus]       [autofocus]
2       [battery type]         [battery]
3              [brand]           [brand]
4       [image format]    [image format]
5                   []  [af illuminator]
6                   []  [exposure modes]
7              [focus]                []
8                   []    [product name]

"""

# Il problema è che in auto-focus e product non ho i valori che mi servono.
# Sono costretto a cambiare l'algoritmo per la generazione delle corrispondenze.

"""
Tentativo n.3 -> (Il gioco èp fatto, un saluto)

==================================================
[{'value_overlap': 'SJ'}, {'value_overlap': 'EXT_JAC_LEV'}, {'label': 'JAC'}]
   MT  TP  FP  FN    P    R    F
0  20  20   0   0  1.0  1.0  1.0
SOURCE                   S1                     S2
GAT                                               
1       [auto focus, focus]            [autofocus]
2            [battery type]              [battery]
3                   [brand]  [product name, brand]
4            [image format]         [image format]
5                        []       [af illuminator]
6                        []       [exposure modes]
==================================================
[{'label': 'LEV'}, {'value_overlap': 'JAC'}, {'label': 'OC'}]
   MT  TP  FP  FN    P    R    F
0  20  20   0   0  1.0  1.0  1.0
SOURCE                   S1                     S2
GAT                                               
1       [auto focus, focus]            [autofocus]
2            [battery type]              [battery]
3                   [brand]  [brand, product name]
4            [image format]         [image format]
5                        []       [af illuminator]
6                        []       [exposure modes]
==================================================
[{'label': 'JARO'}, {'value_overlap': 'GEN_JAC'}, {'value_overlap': 'SJ'}]
   MT  TP  FP  FN       P    R       F
0  22  20   2   0  0.9091  1.0  0.9524
SOURCE                   S1                              S2
GAT                                                        
1       [auto focus, focus]                     [autofocus]
2            [battery type]                       [battery]
3                   [brand]           [brand, product name]
4            [image format]  [image format, af illuminator]
5                        []                [exposure modes]

"""
exit(1)
