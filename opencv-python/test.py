#!/bin/python3

def foo(a, b, c):
    print(a,b,c)


ab = (1, 2)
c = 3
foo(*ab, c)