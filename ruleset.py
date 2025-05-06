class RuleSet:
    def __init__(self, fn):
        self.fn = fn

    def is_member(self, node):
        return self.fn(node)

class FiniteSet(RuleSet):
    def __init__(self, s:set):
        self.s = s
        super().__init__(lambda node: node in self.s)

class Negation(RuleSet):
    def __init__(self, r1:RuleSet):
        super().__init__(lambda node: not r1.is_member(node))

class Disjunction(RuleSet):
    def __init__(self):
        super().__init__(self.fn)
        self.r_sets = []

    def add_rule_set(self, r_set:RuleSet):
        self.r_sets.append(r_set)

    def fn(self, node):
        result = False
        for r_set in self.r_sets:
            result = result or r_set.is_member(node)
            if result:
                break
        return result

class Conjunction(RuleSet):
    def __init__(self):
        super().__init__(self.fn)
        self.r_sets = []

    def add_rule_set(self, r_set:RuleSet):
        self.r_sets.append(r_set)

    def fn(self, node):
        result = True
        for r_set in self.r_sets:
            result = result and r_set.is_member(node)
            if not result:
                break
        return result

c = AndCollection()
c.add_rule_set(RuleSet(lambda x: not bool(x % 2)))
c.add_rule_set(RuleSet(lambda x: not bool(x % 3)))



for n in range(0, 13):
    print(c.is_member(n))


