"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Iterable, Callable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """f(x,y) = x * y"""
    return x * y


def id(x: float) -> float:
    """f(x) = x"""
    return x


def add(x: float, y: float) -> float:
    """f(x, y) = x + y"""
    return x + y


def neg(x: float) -> float:
    """Return negative number"""
    return -x


def lt(x: float, y: float) -> float:
    """Return true if x less than y"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return true if x is equal to y"""
    return abs(x - y) <= 1e-8


def max(x: float, y: float) -> float:
    """Return maximum of x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return true if x is close to y"""
    return abs(x - y) <= 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid"""
    if x >= 0:
        return 1.0 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """f(x) = x if x is greater than 0, else 0"""
    if x > 0:
        return x
    return 0


EPS = 1e-6


def log(x: float) -> float:
    """Return log of x"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Return exponential of x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Return inverse of x"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Return backprop of log of x times d"""
    return d * inv(x)


def inv_back(x: float, d: float) -> float:
    """Return derivative of inverse of x times d"""
    return -d * inv(pow(x, 2))


def relu_back(x: float, d: float) -> float:
    """Return derivative of relu times d"""
    if x > 0:
        return d
    else:
        return 0


def addLists(items1: Iterable[float], items2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists"""
    return zipWith(add, items1, items2)


def negList(items: Iterable[float]) -> Iterable[float]:
    """Negate all elements"""
    return map(neg, items)


def prod(items: Iterable[float]) -> float:
    """Calculate the product of all elements in a list"""
    return reduce(mul, items)


def sum(items: Iterable[float]) -> float:
    """Sum all elements in a list"""
    return reduce(add, items)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable, items: Iterable[float]) -> Iterable[float]:
    """Apply func to each element in items list"""
    ret = []
    for item in items:
        ret.append(func(item))
    return ret


def zipWith(
    func: Callable, list1: Iterable[float], list2: Iterable[float]
) -> Iterable[float]:
    """Combine elements from two iterables using a func"""
    ret = []
    for a, b in zip(list1, list2):
        ret.append(func(a, b))
    return ret


def reduce(func: Callable, items: Iterable[float]) -> float:
    """Reduce an iterable to a single value using a given func"""
    item_list = list(items)
    if len(item_list) == 0:
        return 0
    elif len(item_list) == 1:
        return item_list[0]

    ret = func(item_list[0], item_list[1])
    for i in range(2, len(item_list)):
        ret = func(ret, item_list[i])
    return ret
