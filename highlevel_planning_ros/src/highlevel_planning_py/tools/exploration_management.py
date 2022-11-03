from highlevel_planning_py.execution.es_sequential_execution import (
    execute_plan_sequentially,
)


def print_plan(sequence, parameters):
    print("---------------------------------------------------")
    print("Found plan:")
    for idx, seq_item in enumerate(sequence):
        print(f"{seq_item} {parameters[idx]}")
    print("---------------------------------------------------")


def run_exploration(
    world, kb, rep, args, get_execution_system, xplorer, demo_sequence, demo_parameters
):
    # Store initial state
    initial_state_id = world.save_state()

    explored = False
    while True:
        # Plan
        plan = kb.solve()
        rep.report_after_planning(plan, kb)

        # Execute
        if plan is False:
            planning_failed = True
            print("No plan found.")
        else:
            planning_failed = False
            sequence, parameters = plan
            print_plan(sequence, parameters)
            if not args.noninteractive:
                input("Press enter to run...")
            es = get_execution_system(sequence, parameters)
            res = execute_plan_sequentially(es, verbose=True)
            rep.report_after_execution(res)
            if res:
                print("Reached goal successfully. Exiting.")
                break
            else:
                print("Failure during plan execution.")

        if explored:
            print("Already explored once, aborting.")
            break

        # Decide what happens next
        if not args.noninteractive:
            choice = input(f"Choose next action: (a)bort, (e)xplore\nYour choice: ")
        else:
            choice = "e"
        if choice == "e":
            # Exploration
            explored = True
            rep.report_before_exploration(kb, plan)
            success, metrics = xplorer.exploration(
                planning_failed,
                demo_sequence,
                demo_parameters,
                state_id=initial_state_id,
                no_seed=args.no_seed,
            )
            rep.report_after_exploration(kb, metrics)
            if not success:
                print("Exploration was not successful")
                break
        else:
            if choice != "a":
                print("Invalid choice, aborting.")
            break
