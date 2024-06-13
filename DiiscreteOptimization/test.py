def f(x):
    return x / (x ** 2 + 1)

def g(x, r):
    return x - r * (x ** 2 + 1)

def gradient(x, r):
    return 1 - 2 * r * x

x = 100
r = f(x)
for _ in range(15):
    x = 1 / (r * 2)
    r = f(x)
    print(r)