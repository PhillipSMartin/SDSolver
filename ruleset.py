from typing import Any, Callable


class RuleSet:
    def __init__(self, fn:Callable[[Any],bool] = None, descr:str = None):
        self.fn = fn
        self.s = None
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

class IdSet(RuleSet):
    def __init__(self, s:set = None, descr:str = None):
        super().__init__(self.fn, descr=descr)
        self.s = s

    def fn(self, node)->bool:
        return node.id() in self.s

    def __str__(self):
        return self.descr or f'{self.s}'


class Negation(RuleSet):
    def __init__(self, r:RuleSet, descr: str = None):
        self.r = r
        super().__init__(self.fn, descr=descr)

    def fn(self, node)->bool:
        return not node in self.r

    def __str__(self):
        return self.descr or f"Not {self.r}"


class Disjunction(RuleSet):
    def __init__(self, *args, descr: str = None):
        super().__init__(self.fn, descr = descr)
        self.r_sets = []
        self.s = set() # the disjunction of all IdSets added
        for r in args:
            self.add_rule_set(r)

    def add_rule_set(self, r_set: RuleSet):
        self.r_sets.append(r_set)
        if r_set.s is not None:
            self.s |= r_set.s

    def __ior__(self, other:RuleSet):
        self.add_rule_set(other)
        return self

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
        super().__init__(self.fn, descr=descr)
        self.r_sets = []
        self.s = None # the conjunction of all IdSets added
        for r in args:
            self.add_rule_set(r)

    def add_rule_set(self, r_set: RuleSet):
        self.r_sets.append(r_set)
        if r_set.s is not None:
            if self.s is None:
                self.s = r_set.s
            else:
                self.s &= r_set.s

    def __ior__(self, other:RuleSet):
        self.add_rule_set(other)
        return self

    def fn(self, node):
        result = True
        for r_set in self.r_sets:
            result = result and node in r_set
            if not result:
                break
        return result

    def __str__(self):
        return self.descr or "(" + " and ".join([str(r_set) for r_set in self.r_sets]) + ")"

# b = IdSet({2, 3})
# c = RuleSet(lambda n: n > 5)
# a = Conjunction(b, c)
# print(a.s)
