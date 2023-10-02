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
  Match = Match[['A', 'B']]
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

