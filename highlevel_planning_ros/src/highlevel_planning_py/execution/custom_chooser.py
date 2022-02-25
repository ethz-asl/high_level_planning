from py_trees.composites import Selector
from py_trees.common import Status


class CustomChooser(Selector):
    """
    Choosers are Selectors with Commitment
    .. graphviz:: dot/chooser.dot
    A variant of the selector class. Once a child is selected, it
    cannot be interrupted by higher priority siblings. As soon as the chosen child
    itself has finished it frees the chooser for an alternative selection. i.e. priorities
    only come into effect if the chooser wasn't running in the previous tick.
    .. note::
        This is the only composite in py_trees that is not a core composite in most behaviour tree implementations.
        Nonetheless, this is useful in fields like robotics, where you have to ensure that your manipulator doesn't
        drop it's payload mid-motion as soon as a higher interrupt arrives. Use this composite
        sparingly and only if you can't find another way to easily create an elegant tree composition for your task.
    Args:
        name (:obj:`str`): the composite behaviour name
        children ([:class:`~py_trees.behaviour.Behaviour`]): list of children to add
    """

    def __init__(self, name="Chooser", children=None):
        super(CustomChooser, self).__init__(name, children)

    def tick(self):
        """
        Run the tick behaviour for this chooser. Note that the status
        of the tick is (for now) always determined by its children, not
        by the user customised update function.
        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # Required behaviour for *all* behaviours and composites is
        # for tick() to check if it isn't running and initialise
        if self.status != Status.RUNNING:
            # chooser specific initialisation
            # invalidate everything
            for child in self.children:
                child.stop(Status.INVALID)
            self.current_child = None
            # run subclass (user) initialisation
            self.initialise()
        # run any work designated by a customised instance of this class
        self.update()
        if self.current_child is not None:
            # run our child, and invalidate anyone else who may have been ticked last run
            # (bit wasteful always checking for the latter)
            previous = self.current_child
            passed = False
            found_running_or_success = False
            for child in self.children:
                if child is self.current_child:
                    passed = True
                elif child.status != Status.INVALID and not passed:
                    child.stop(Status.INVALID)
                if passed:
                    for node in child.tick():
                        yield node
                        if node is child:
                            if (
                                node.status == Status.RUNNING
                                or node.status == Status.SUCCESS
                            ):
                                self.current_child = child
                                if previous != self.current_child:
                                    passed2 = False
                                    for child2 in self.children:
                                        if passed2:
                                            if child2.status != Status.INVALID:
                                                child2.stop(Status.INVALID)
                                        passed2 = (
                                            True
                                            if child2 == self.current_child
                                            else passed2
                                        )
                                found_running_or_success = True
                                break
                    if found_running_or_success:
                        break
            if not found_running_or_success:
                # Ran out of children
                self.current_child = None
        else:
            for child in self.children:
                for node in child.tick():
                    yield node
                if child.status == Status.RUNNING or child.status == Status.SUCCESS:
                    self.current_child = child
                    break
        new_status = (
            self.current_child.status
            if self.current_child is not None
            else Status.FAILURE
        )
        self.stop(new_status)
        yield self
