import string, re

def string_pre_process(s: str, char: str = string.punctuation):
    """
    String pre-processing applying all 'good habits' rules.
    - Convert string to lowercase
    - Remove all punctual characters
    :param s: string to process
    :param char: all punctual characters provided by python string class
    :return: processed string ready to be overwritten inside dataframe cols
    """
    if type(s) is str:
        s = s.lower()
        for c in char:
            s = s.replace(c, " ")
    else:
        s = str(s)
    s = re.sub(" +", " ", s)
    return s.strip()
