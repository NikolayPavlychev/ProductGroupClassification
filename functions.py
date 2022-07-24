def voc_filter(x):
    import re
    x_clear = re.sub(r'\w*\d\w*', '', x).strip()
    return x_clear

def lemmatizer(x):
    x_norm = []
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    for i in x:
        x_norm.append((morph.parse(i)[0]).normal_form)
    return x_norm