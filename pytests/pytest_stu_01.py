#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/1 21:15
import pytest


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def test_add():
    assert add(1, 2) == 3
    assert add(2, 2) == 4
    assert add(3, 1) == 4


@pytest.mark.parametrize("a, b, expected", [
    (1, 1, 2),
    (-1, 1, 0),
])
def test_operate_add(a, b, expected):
    assert add(a, b) == expected


@pytest.mark.parametrize("a, b, expected", [
    (5, 3, 2),
    (5, 2, 3),
    (5, 1, 4),
])
def test_operate_subtract(a, b, expected):
    assert subtract(a, b) == expected
