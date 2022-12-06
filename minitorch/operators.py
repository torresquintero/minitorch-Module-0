"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Any, Callable, Iterable, List

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    return -x


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> Any:
    # see https://github.com/python/typeshed/issues/7733 for why typing is Any
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.e ** (-x))
    return math.e**x / (1.0 + math.e**x)


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    return d * (1 / x)


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    return d * (-1 / x**2)


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    if x <= 0:
        return 0.0
    return d * 1.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def result(input_list: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in input_list]

    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    negator = map(neg)
    return negator(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], List[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def result(ls1: Iterable[float], ls2: List[float]) -> Iterable[float]:
        new_list = [fn(x, ls2[i]) for i, x in enumerate(ls1)]
        return new_list

    return result


def addLists(ls1: Iterable[float], ls2: List[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    adder = zipWith(add)
    return adder(ls1, ls2)


def reduce(fn: Callable[[float, float], float]) -> Callable[[List[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def result(ls: List[float]) -> Any:
        def recursive_function(i: int, ls: List[float]) -> Any:
            if i > 0:
                return fn(ls[i], recursive_function(i - 1, ls))
            return ls[i]

        if ls == []:
            return 0
        return recursive_function(len(ls) - 1, ls)

    return result


def summation(ls: List[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    adder = reduce(add)
    return adder(ls)


def prod(ls: List[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    multiplier = reduce(mul)
    return multiplier(ls)
