class PredicatesPDDL:
    def __init__(self, world):
        self.world = world
        self.call = dict()
        self.descriptions = dict()

    def register_predicate(self, predicate_name, predicate_description):
        self.call[predicate_name] = lambda *args: self.check_predicate(
            predicate_name, *args
        )
        self.descriptions[predicate_name] = [
            [name.strip("?"), pred_type]
            for name, pred_type in predicate_description.items()
        ]

    def check_predicate(self, predicate_name, *args):
        pred_to_check = (predicate_name, *args)
        return self.world.check_predicate(pred_to_check)
