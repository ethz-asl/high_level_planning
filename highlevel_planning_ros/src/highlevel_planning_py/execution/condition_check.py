import py_trees.common
import multiprocessing
import atexit
import time


class ConditionChecker_Blackboard(py_trees.behaviour.Behaviour):
    def __init__(self, checker_variable):
        super(ConditionChecker_Blackboard, self).__init__(
            name="checker_" + checker_variable
        )
        self._checker_variable = checker_variable
        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self):
        val = self.blackboard.get(self._checker_variable)
        if val:
            new_status = py_trees.common.Status.SUCCESS
            self.feedback_message = "Check succeeded"
        else:
            new_status = py_trees.common.Status.FAILURE
            self.feedback_message = "Check failed"
        self.logger.debug(
            "%s.update()[%s->%s][%s]"
            % (self.__class__.__name__, self.status, new_status, self.feedback_message)
        )
        return new_status


class ConditionChecker_Predicate(py_trees.behaviour.Behaviour):
    def __init__(self, predicate_fcn, predicate_args, invert=False):
        invert_str = ""
        if invert:
            invert_str = "_inverted"
        super(ConditionChecker_Predicate, self).__init__(
            name="checker_pred_" + predicate_fcn.__name__ + invert_str
        )
        self.setup_called = False
        self._predicate_fcn = predicate_fcn
        self._invert = invert
        self._predicate_args = predicate_args

    def setup(self, unused_timeout=15):
        if not self.setup_called:
            self.parent_connection, self.child_connection = multiprocessing.Pipe()
            self.check_process = multiprocessing.Process(
                target=checker_process,
                args=(self.child_connection, self._predicate_fcn, self._predicate_args),
            )
            atexit.register(self.check_process.terminate)
            self.check_process.start()
            self.setup_called = True
        return True

    def initialise(self):
        self.logger.debug(
            "%s.initialise()->initiating check" % (self.__class__.__name__)
        )
        if not self.setup_called:
            raise RuntimeError("Setup function not called")

        # Clear out any old responses from the pipe
        while self.parent_connection.poll():
            _ = self.parent_connection.recv()

        # Send command
        # if not self.parent_connection.poll():
        self.parent_connection.send(["start"])

    def update(self):
        # print("child process status: "+str(self.check_process.is_alive()))
        if self.parent_connection.poll():
            res = self.parent_connection.recv()
            if self._invert:
                res = not res
            if res:
                new_status = py_trees.common.Status.SUCCESS
                self.feedback_message = "Check succeeded"
            else:
                new_status = py_trees.common.Status.FAILURE
                self.feedback_message = "Check failed"
        else:
            new_status = py_trees.common.Status.RUNNING
            self.feedback_message = "Check in progress"
        self.logger.debug(
            "%s.update()[%s->%s][%s]"
            % (self.__class__.__name__, self.status, new_status, self.feedback_message)
        )
        return new_status


def checker_process(pipe_connection, fcn, fcn_args):
    while True:
        if pipe_connection.poll():
            # print("got check command: "+fcn.__name__)
            _ = pipe_connection.recv()
            res = fcn(*fcn_args)
            pipe_connection.send(res)
            # print("Response sent: "+fcn.__name__)
        time.sleep(0.5)
