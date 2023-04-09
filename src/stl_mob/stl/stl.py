from collections import deque
from typing import Union

import numpy as np
from stlpy.STL import STLTree, STLFormula, LinearPredicate

COLORED = False

if COLORED:
    from termcolor import colored
else:
    def colored(text, color):
        return text


class STL:
    def __init__(self, ast: Union[list, str, STLTree, STLFormula, LinearPredicate]):
        self.ast = ast
        self.single_operators = ("~", "G", "F")
        self.binary_operators = ("&", "|", "->", "U")
        self.sequence_operators = ("G", "F", "U")
        self.stlpy_form = None
        self.expr_repr = None

    """
    Syntax Functions
    """

    def __and__(self, other: 'STL') -> 'STL':
        ast = ["&", self.ast, other.ast]
        return STL(ast)

    def __or__(self, other: 'STL') -> 'STL':
        ast = ["|", self.ast, other.ast]
        return STL(ast)

    def __invert__(self) -> 'STL':
        ast = ["~", self.ast]
        return STL(ast)

    def implies(self, other: 'STL') -> 'STL':
        ast = ["->", self.ast, other.ast]
        return STL(ast)

    def eventually(self, start: int = 0, end: int = None):
        ast = ["F", self.ast, start, end]
        return STL(ast)

    def always(self, start: int = 0, end: int = None) -> 'STL':
        ast = ["G", self.ast, start, end]
        return STL(ast)

    def until(self, other: 'STL', start: int = 0, end: int = None) -> 'STL':
        ast = ["U", self.ast, other.ast, start, end]
        return STL(ast)

    def get_stlpy_form(self):
        # catch already converted form
        if self.stlpy_form is None:
            self.stlpy_form = self._to_stlpy(self.ast)

        return self.stlpy_form

    def _to_stlpy(self, ast) -> STLTree:
        if self._is_leaf(ast):
            if isinstance(ast, str):
                raise ValueError(f"str variable {ast} not supported")
            self.stlpy_form = ast
            return ast

        if ast[0] == "~":
            self.stlpy_form = self._handle_not(ast)
        elif ast[0] == "G":
            self.stlpy_form = self._handle_always(ast)
        elif ast[0] == "F":
            self.stlpy_form = self._handle_eventually(ast)
        elif ast[0] == "&":
            self.stlpy_form = self._handle_and(ast)
        elif ast[0] == "|":
            self.stlpy_form = self._handle_or(ast)
        elif ast[0] == "->":
            self.stlpy_form = self._handle_implies(ast)
        elif ast[0] == "U":
            self.stlpy_form = self._handle_until(ast)
        else:
            raise ValueError(f"Unknown operator {ast[0]}")

        return self.stlpy_form

    def _handle_not(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.negation()

    def _handle_and(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1 & sub_form_2

    def _handle_or(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1 | sub_form_2

    def _handle_implies(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1.negation() | sub_form_2

    def _handle_eventually(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.eventually(ast[2], ast[3])

    def _handle_always(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.always(ast[2], ast[3])

    def _handle_until(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1.until(sub_form_2, ast[3], ast[4])

    @staticmethod
    def _is_leaf(ast):
        return issubclass(type(ast), STLFormula) or isinstance(ast, str)

    def simplify(self):
        if self.stlpy_form is None:
            self.get_stlpy_form()
        self.stlpy_form.simplify()

    def __repr__(self):
        if self.expr_repr is not None:
            return self.expr_repr

        single_operators = ("~", "G", "F")
        binary_operators = ("&", "|", "->", "U")
        time_bounded_operators = ("G", "F", "U")

        # traverse ast
        operator_stack = [self.ast]
        expr = ""
        cur = self.ast

        def push_stack(ast):
            if isinstance(ast, str) and ast in time_bounded_operators:
                time_window = f"[{cur[-2]}, {cur[-1]}]"
                operator_stack.append(time_window)
            operator_stack.append(ast)

        while operator_stack:
            cur = operator_stack.pop()
            if self._is_leaf(cur):
                expr += cur.__str__()
            elif isinstance(cur, str):
                if cur == "(" or cur == ")":
                    expr += cur
                elif cur.startswith("["):
                    expr += colored(cur, "yellow") + " "
                else:
                    if cur in ("G", "F"):
                        if cur == "F":
                            expr += colored("F", "magenta")
                        else:
                            expr += colored(cur, "magenta")
                    elif cur in ("&", "|", "->", "U"):
                        expr += " " + colored(cur, "magenta")
                        if cur != "U":
                            expr += " "
                    elif cur in ("~",):
                        expr += colored(cur, "magenta")
            elif cur[0] in single_operators:
                # single operator
                if not self._is_leaf(cur[1]):
                    push_stack(")")
                push_stack(cur[1])
                if not self._is_leaf(cur[1]):
                    push_stack("(")
                push_stack(cur[0])
            elif cur[0] in binary_operators:
                # binary operator
                if not self._is_leaf(cur[2]) and cur[2][0] in binary_operators:
                    push_stack(")")
                    push_stack(cur[2])
                    push_stack("(")
                else:
                    push_stack(cur[2])
                push_stack(cur[0])
                if not self._is_leaf(cur[1]) and cur[1][0] in binary_operators:
                    push_stack(")")
                    push_stack(cur[1])
                    push_stack("(")
                else:
                    push_stack(cur[1])

        self.expr_repr = expr
        return expr

    def get_all_predicates(self):
        all_preds = []
        queue = deque([self.ast])

        while queue:
            cur = queue.popleft()

            if self._is_leaf(cur):
                all_preds.append(cur)
            elif cur[0] in self.single_operators:
                queue.append(cur[1])
            elif cur[0] in self.binary_operators:
                queue.append(cur[1])
                queue.append(cur[2])
            else:
                raise RuntimeError("Should never visit here")

        return all_preds


def inside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being inside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return inside_rectangle:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d));
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_min)
    left = LinearPredicate(-a1, -y1_max)

    a2 = np.zeros((1, d));
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_min)
    bottom = LinearPredicate(-a2, -y2_max)

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        inside_rectangle.__str__ = lambda: name
        inside_rectangle.__repr__ = lambda: name

    return inside_rectangle


def outside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_max)
    left = LinearPredicate(-a1, -y1_min)

    a2 = np.zeros((1, d))
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_max)
    bottom = LinearPredicate(-a2, -y2_min)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        outside_rectangle.__str__ = lambda: name
        outside_rectangle.__repr__ = lambda: name

    return outside_rectangle
