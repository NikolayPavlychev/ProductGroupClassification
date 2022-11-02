
def voc_filter(x):
    
    import re

    x_clear = re.split(r'\w*\d\w*|\b\w\w\b|\b\w\b',x)
    x_clear = ' '.join(x_clear)
    x_clear = re.sub(r'\s+',' ',x_clear).strip()

    return x_clear


def lemmatizer(x):
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()

    x_norm = []    
    for i in x:
        x_norm.append((morph.parse(i)[0]).normal_form)
    return x_norm




