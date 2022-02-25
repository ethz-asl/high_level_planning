from abc import ABCMeta, abstractmethod


class ExecutionSystem:
    """
    Abstract base class for execution system.
    """

    __metaclass__ = ABCMeta

    def setup(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def print_status(self):
        pass
