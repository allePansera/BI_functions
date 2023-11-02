import pandas as pd


def toA_B(Y):
    X = Y[['GAT', 'SLAT']].rename(columns={'GAT': 'A','SLAT': 'B'})
    return X


def valuta(gold_schema: pd.DataFrame, global_match: pd.DataFrame):

    Match = global_match[['A', 'B']]
    FOJ = gold_schema.merge(Match, how='outer', indicator=True)

    TP = FOJ[FOJ['_merge'] == 'both']
    FP = FOJ[FOJ['_merge'] == 'right_only']
    FN = FOJ[FOJ['_merge'] == 'left_only']

    if len(TP) == 0:
      return pd.DataFrame({
        'MT': [len(Match)],
        'TP': [len(TP)],
        'FP': [len(FP)],
        'FN': [len(FN)],
        'P': [round(0, 4)],
        'R': [round(0, 4)],
        'F': [round(0, 4)]
      })
    else:
      P = len(TP) / (len(TP) + len(FP))
      R = len(TP) / (len(TP) + len(FN))
      F = 2 * P * R / (P + R)
      return pd.DataFrame({
        'MT': [len(Match)],
        'TP': [len(TP)],
        'FP': [len(FP)],
        'FN': [len(FN)],
        'P': [round(P, 4)],
        'R': [round(R, 4)],
        'F': [round(F, 4)]
      })


def vedi_valuta(Gold: pd.DataFrame, Match: pd.DataFrame, metrics: str):
    #Match = Match[['A', 'B']]
    FOJ = pd.merge(Gold, Match, how='outer', indicator=True)

    TP = FOJ[FOJ['_merge'] == 'both']
    FP = FOJ[FOJ['_merge'] == 'right_only']
    FN = FOJ[FOJ['_merge'] == 'left_only']

    if metrics == 'FP':
        return FP
    if metrics == 'TP':
        return TP
    if metrics == 'FN':
        return FN


def valuta_blocking(DA,DB,Block,Gold):

    Gold.columns=['l_id','r_id']
    Block=Block[['l_id','r_id']]
    JOIN=pd.merge(Gold, Block)

    #Reduction_Ratio
    RR=1-len(Block)/(len(DA)*len(DB))
    #Pairs Completeness o Recall
    PC = len(JOIN)/len(Gold)
    #Pairs Quality
    PQ = len(JOIN)/len(Block)
    risultato = pd.DataFrame([(DA.shape[0],DB.shape[0],Block.shape[0],round(RR,4),round(PC,4),round(PQ,4))],
                             columns=['A', 'B', 'BlockSize', 'ReductRatio', 'PCompletness', 'PQuality'])

    return risultato

