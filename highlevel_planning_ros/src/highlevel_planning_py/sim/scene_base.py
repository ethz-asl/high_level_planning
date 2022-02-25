from copy import deepcopy
import pybullet as pb


class SceneBase:
    def __init__(self, world, asset_dir, restored_objects=None):
        self.world = world
        self.objects = dict()
        self.asset_dir = asset_dir

        if restored_objects is not None:
            self.objects = restored_objects

    def set_objects(self, objects):
        self.objects = deepcopy(objects)

    def add_objects(self, force_load=False):
        for key, obj in self.objects.items():
            if self.objects[key].model is None or force_load:
                self.objects[key].model = self.world.add_model(
                    obj.urdf_path, obj.init_pos, obj.init_orient, scale=obj.scale
                )
            if self.objects[key].friction_setting is not None:
                for spec in self.objects[key].friction_setting:
                    pb.changeDynamics(
                        self.objects[key].model.uid,
                        self.objects[key].model.link_name_to_index[spec["link_name"]],
                        lateralFriction=spec["lateral_friction"],
                        physicsClientId=self.world.client_id,
                    )
            if self.objects[key].joint_setting is not None:
                for spec in self.objects[key].joint_setting:
                    pb.setJointMotorControl2(
                        self.objects[key].model.uid,
                        spec["jnt_idx"],
                        controlMode=spec["mode"],
                        force=spec["force"],
                        physicsClientId=self.world.client_id,
                    )
