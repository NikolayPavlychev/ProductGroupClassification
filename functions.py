from functools import lru_cache


def voc_filter(x):
    
    import re
    # x_clear = re.sub(r'\w*\d\w*', '', x).strip()
    x_clear = re.split(r'\w*\d\w*|\b\w\w\b|\b\w\b',x)
    x_clear = ' '.join(x_clear)
    x_clear = re.sub(r'\s+',' ',x_clear).strip()
    
    # x_clear = re.sub(r' \w\w ', ' ', x).strip()
    # x_clear = re.sub(r'  ', ' ', x).strip()
    return x_clear


# @lru_cache(maxsize=None) 
def lemmatizer(x):
    print(x)
    
    x_norm = []
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    
    for i in x:
        x_norm.append((morph.parse(i)[0]).normal_form)
    return x_norm




