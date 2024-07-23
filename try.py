from functools import partial
def plus(a, b, c, d, e):
    return a+b+c+d+e

dic1 = {'c':2, 'b':3}
dic2 = {'d':4, 'e':5}
plus_p = partial(plus, **dic1, **dic2)
print(plus_p(1))