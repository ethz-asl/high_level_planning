from pddl_parser.PDDL import PDDL_Parser


class CustomPDDLParser(PDDL_Parser):
    def __init__(self, warn_only_once=True):
        self.warn_only_once = warn_only_once
        self.already_warned_about = set()

        self.ignored_requirements = [":hierarchy", ":method-preconditions"]

    def parse_domain(self, domain_filename, requirements=None):
        requirements = PDDL_Parser.SUPPORTED_REQUIREMENTS + self.ignored_requirements
        super().parse_domain(domain_filename, requirements)

    def parse_domain_extended(self, t, group):
        if self.warn_only_once and t in self.already_warned_about:
            return
        super().parse_domain_extended(t, group)
        self.already_warned_about.add(t)
