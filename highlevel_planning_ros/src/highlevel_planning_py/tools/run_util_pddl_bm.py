from highlevel_planning_py.knowledge.knowledge_base import KnowledgeBase
from highlevel_planning_py.sim_pddl.predicates import PredicatesPDDL


def setup_knowledge_base_pddl_bm(paths, parser_init, args, goals, world):
    # Knowledge base
    kb = KnowledgeBase(paths, parser_init.domain_name, domain_file=args.domain_file)
    kb.set_goals(goals)

    # Extract actions in correct format
    for action in parser_init.actions:
        params = [[param[0].strip("?"), param[1]] for param in action.parameters]
        preconds_pos = [
            (precond[0], True, [param.strip("?") for param in precond[1:]])
            for precond in action.positive_preconditions
        ]
        preconds_neg = [
            (precond[0], False, [param.strip("?") for param in precond[1:]])
            for precond in action.negative_preconditions
        ]
        preconds = preconds_pos + preconds_neg
        effects_pos = [
            (effect[0], True, [param.strip("?") for param in effect[1:]])
            for effect in action.add_effects
        ]
        effects_neg = [
            (effect[0], False, [param.strip("?") for param in effect[1:]])
            for effect in action.del_effects
        ]
        effects = effects_pos + effects_neg
        action_descr = {
            "params": params,
            "preconds": preconds,
            "effects": effects,
            "exec_ignore_effects": list(),
        }
        kb.add_action(action.name, action_descr, overwrite=True)

    # Add types
    for parent_label, child_labels in parser_init.types.items():
        kb.add_type(parent_label)
        for child_label in child_labels:
            kb.add_type(child_label, parent_label)

    # Add predicates
    preds = PredicatesPDDL(world)
    for pred, descr in parser_init.predicates.items():
        preds.register_predicate(pred, descr)
        kb.add_predicate(
            predicate_name=pred,
            predicate_definition=preds.descriptions[pred],
            overwrite=True,
        )
    kb.set_predicate_funcs(preds)

    # Add objects
    for type_label, objects in parser_init.objects.items():
        for obj in objects:
            kb.add_object(obj, type_label)

    # Set initial predicates
    for pred_tuple in parser_init.state:
        kb.initial_state_predicates.add(pred_tuple)

    return kb, preds
