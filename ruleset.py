from typing import Callable


class RuleSet:
    def __init__(self, fn: Callable[[str], bool] = None, descr: str = None):
        self.fn = fn
        self.descr = descr

    def contains(self, node):
        return self.fn(node) if self.fn else False

    def __str__(self):
        if self.descr:
            return self.descr
        elif not self.fn:
            return "empty set"
        else:
            return "RuleSet"


class EnumeratedSet(RuleSet):
    def __init__(self, s: set = None):
        self.s = s or set()
        super().__init__(lambda node: node in self.s)

    def __str__(self):
        return str(self.s)


class Negation(RuleSet):
    def __init__(self, r: RuleSet):
        self.r = r
        super().__init__(lambda node: not r.contains(node))

    def __str__(self):
        return f"Not {self.r}"


class Disjunction(RuleSet):
    def __init__(self, *args):
        super().__init__(self.fn)
        self.r_sets = list(args)

    def add_rule_set(self, r_set: RuleSet):
        self.r_sets.append(r_set)

    def fn(self, node):
        result = False
        for r_set in self.r_sets:
            result = result or r_set.contains(node)
            if result:
                break
        return result

    def __str__(self):
        return " or ".join([str(r_set) for r_set in self.r_sets])


class Conjunction(RuleSet):
    def __init__(self, *args):
        super().__init__(self.fn)
        self.r_sets = list(args)

    def add_rule_set(self, r_set: RuleSet):
        self.r_sets.append(r_set)

    def fn(self, node):
        result = True
        for r_set in self.r_sets:
            result = result and r_set.contains(node)
            if not result:
                break
        return result

    def __str__(self):
        return " and ".join([str(r_set) for r_set in self.r_sets])

