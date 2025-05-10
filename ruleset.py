from typing import Any, Callable


class RuleSet:
    def __init__(self, fn:Callable[[Any],bool] = None, descr: str = None):
        self.fn = fn
        self.descr = descr

    def __contains__(self, item):
        return self.fn(item) if self.fn else False

    def __str__(self):
        if self.descr:
            return self.descr
        elif not self.fn:
            return "empty set"
        else:
            return "RuleSet"


class EnumeratedSet(RuleSet):
    def __init__(self, s: set = None, descr: str = None):
        self.s = s or set()
        super().__init__(lambda node: node in self.s, descr = descr)

    def __str__(self):
        return self.descr or str(self.s)


class Negation(RuleSet):
    def __init__(self, r: RuleSet, descr: str = None):
        self.r = r
        super().__init__(lambda node: not node in r, descr = descr)

    def __str__(self):
        return self.descr or f"Not {self.r}"


class Disjunction(RuleSet):
    def __init__(self, *args, descr: str = None):
        super().__init__(self.fn, descr = descr)
        self.r_sets = list(args)

    def add_rule_set(self, r_set: RuleSet):
        self.r_sets.append(r_set)

    def fn(self, node):
        result = False
        for r_set in self.r_sets:
            result = result or node in r_set
            if result:
                break
        return result

    def __str__(self):
       return self.descr or "(" + " or ".join([str(r_set) for r_set in self.r_sets]) + ")"


class Conjunction(RuleSet):
    def __init__(self, *args, descr: str = None):
        super().__init__(self.fn, descr = descr)
        self.r_sets = list(args)

    def add_rule_set(self, r_set: RuleSet):
        self.r_sets.append(r_set)

    def fn(self, node):
        result = True
        for r_set in self.r_sets:
            result = result and node in r_set
            if not result:
                break
        return result

    def __str__(self):
        return self.descr or "(" + " and ".join([str(r_set) for r_set in self.r_sets]) + ")"

