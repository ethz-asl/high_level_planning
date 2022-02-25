import pybullet as pb
import os
import pybullet_data
import numpy as np
import time
from math import ceil
import atexit

# from dishwasher_challenge.utils import add_mesh_object


class World(object):
    def __init__(self, sleep=True):
        self.sleep_flag = sleep

        self.sim_time = 0.0
        self.f_s = 240.0
        self.T_s = 1.0 / float(self.f_s)

        self.collision_checker = None
        self.cross_uid = ()

        self.forces = []

        self.colors = {
            "green": (123.0 / 255.0, 159.0 / 255.0, 53.0 / 255.0),
            "red": (170.0 / 255.0, 57.0 / 255.0, 57.0 / 255.0),
            "blue": (34.0 / 255.0, 102.0 / 255.0, 102.0 / 255.0),
            "yellow": (1.0, 1.0, 0.0),
        }

        self.velocity_setter = None
        # atexit.register(self.close)

    def sleep(self, seconds):
        if self.sleep_flag:
            time.sleep(seconds)

    def step_one(self):
        raise NotImplementedError

    def step_seconds(self, secs):
        for _ in range(int(ceil(secs * self.f_s))):
            self.step_one()
            self.sleep(self.T_s)

    def close(self):
        raise NotImplementedError


class _Model:
    def __init__(self, physics_client):
        self._physics_client = physics_client
        self.uid = 0
        self.name = ""
        self.link_name_to_index = dict()

    def load(self, path, position, orientation, scale):
        model_path = os.path.expanduser(path)
        ending = model_path.split(".")[-1]
        if ending == "urdf":
            self.uid = pb.loadURDF(
                model_path,
                position,
                orientation,
                globalScaling=scale,
                physicsClientId=self._physics_client,
            )
        elif ending == "sdf":
            tmp = pb.loadSDF(
                sdfFileName=model_path,
                globalScaling=scale,
                physicsClientId=self._physics_client,
            )
            assert len(tmp) == 1
            self.uid = tmp[0]
            pb.resetBasePositionAndOrientation(
                self.uid,
                posObj=position,
                ornObj=orientation,
                physicsClientId=self._physics_client,
            )
        else:
            self.uid = add_mesh_object(
                model_path, position, orientation, self._physics_client
            )
        self.name = pb.getBodyInfo(self.uid, physicsClientId=self._physics_client)

        for i in range(pb.getNumJoints(self.uid, physicsClientId=self._physics_client)):
            info = pb.getJointInfo(self.uid, i, physicsClientId=self._physics_client)
            name = info[12] if type(info[12]) is str else info[12].decode("utf-8")
            self.link_name_to_index[name] = i

    def remove(self):
        pb.removeBody(self.uid, physicsClientId=self._physics_client)


class WorldPybullet(World):
    def __init__(
        self,
        style="gui",
        sleep=True,
        load_objects=True,
        savedir=None,
        include_floor=True,
    ):
        super(WorldPybullet, self).__init__(sleep)
        self.include_floor = include_floor
        if not load_objects:
            assert savedir is not None

        if style == "gui":
            self.client_id = pb.connect(pb.GUI)
        elif style == "shared":
            self.client_id = pb.connect(pb.SHARED_MEMORY)
        elif style == "direct":
            self.client_id = pb.connect(pb.DIRECT)
        else:
            raise ValueError

        if load_objects:
            pb.resetSimulation(physicsClientId=self.client_id)
        else:
            self.restore_state_file(os.path.join(savedir, "state.bullet"))
            pb.removeAllUserDebugItems(physicsClientId=self.client_id)

        self.basic_settings()

        # Persistence for storing states
        self.active_constraints = list()

    def basic_settings(self):
        pb.configureDebugVisualizer(
            pb.COV_ENABLE_GUI, 0, physicsClientId=self.client_id
        )
        pb.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        if self.include_floor:
            pb.loadURDF("plane.urdf", physicsClientId=self.client_id)

    def add_model(self, path, position, orientation, scale=1.0):
        model = _Model(self.client_id)
        model.load(path, position, orientation, scale)
        return model

    def del_model(self, model):
        model.remove()

    def draw_cross(self, point):
        if len(self.cross_uid) > 0:
            pb.removeUserDebugItem(self.cross_uid[0], physicsClientId=self.client_id)
            pb.removeUserDebugItem(self.cross_uid[1], physicsClientId=self.client_id)
            pb.removeUserDebugItem(self.cross_uid[2], physicsClientId=self.client_id)
        start1 = point - np.array([0.1, 0.0, 0.0])
        end1 = point + np.array([0.1, 0.0, 0.0])
        start2 = point - np.array([0.0, 0.1, 0.0])
        end2 = point + np.array([0.0, 0.1, 0.0])
        start3 = point - np.array([0.0, 0.0, 0.1])
        end3 = point + np.array([0.0, 0.0, 0.1])
        color = np.array([255, 0, 0]) / 255.0
        width = 1.0
        lifetime = 0
        uid1 = pb.addUserDebugLine(start1, end1, color, width, lifetime)
        uid2 = pb.addUserDebugLine(start2, end2, color, width, lifetime)
        uid3 = pb.addUserDebugLine(start3, end3, color, width, lifetime)
        self.cross_uid = (uid1, uid2, uid3)

    def draw_arrow(self, point, direction, color, length=0.2, replace_id=None):
        """ Accepts a point and a direction in world frame and draws it in the simulation """
        tip = point + direction / np.linalg.norm(direction) * length
        color = self.colors[color]
        width = 4.5
        lifetime = 0
        if replace_id is not None:
            arrow_id = pb.addUserDebugLine(
                point.tolist(),
                tip.tolist(),
                color,
                width,
                lifetime,
                replaceItemUniqueId=replace_id,
            )
        else:
            arrow_id = pb.addUserDebugLine(
                point.tolist(), tip.tolist(), color, width, lifetime
            )
        return arrow_id

    def step_one(self):
        for frc in self.forces:
            pb.applyExternalForce(frc[0], frc[1], frc[2], frc[3], frc[4])
        if self.velocity_setter is not None:
            self.velocity_setter()
        pb.stepSimulation(physicsClientId=self.client_id)
        if self.collision_checker is not None:
            self.collision_checker()

    def reset(self):
        pb.resetSimulation(physicsClientId=self.client_id)
        self.basic_settings()

    def close(self):
        pb.disconnect(physicsClientId=self.client_id)

    def restore_state_file(self, filepath):
        pb.restoreState(fileName=filepath, physicsClientId=self.client_id)

    def save_state(self):
        state_id = pb.saveState(physicsClientId=self.client_id)
        constraint_list = [constraint[1] for constraint in self.active_constraints]
        return state_id, constraint_list

    def restore_state(self, saved_state):
        self.delete_all_constraints()
        pb.restoreState(stateId=saved_state[0], physicsClientId=self.client_id)
        for constraint in saved_state[1]:
            self.add_constraint(constraint)

    def add_constraint(self, constraint_spec):
        constraint_id = pb.createConstraint(
            constraint_spec.parent_uid,
            constraint_spec.parent_link_id,
            constraint_spec.child_uid,
            constraint_spec.child_link_id,
            jointType=pb.JOINT_FIXED,
            jointAxis=[1.0, 0.0, 0.0],
            parentFramePosition=constraint_spec.trafo_pos,
            childFramePosition=[0.0, 0.0, 0.0],
            parentFrameOrientation=constraint_spec.trafo_orient,
            physicsClientId=self.client_id,
        )
        self.active_constraints.append((constraint_id, constraint_spec))

    def delete_all_constraints(self):
        for constraint in self.active_constraints:
            pb.removeConstraint(
                userConstraintUniqueId=constraint[0], physicsClientId=self.client_id
            )
        self.active_constraints.clear()
