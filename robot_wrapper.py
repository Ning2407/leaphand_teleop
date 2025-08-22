from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import pinocchio as pin

class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self, urdf_path: str, use_collision=False, use_visual=False, package_dirs: Optional[List[str]]=None):
        # 1) 基础 model/data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if self.model.nv != self.model.nq:
            raise NotImplementedError("Can not handle robot with special joint.")

        self.q0 = pin.neutral(self.model)

        # 2) 可选：加载几何模型（碰撞/可视）
        self.collision_model = None
        self.collision_data = None
        self.visual_model = None
        self.visual_data = None

        # package_dirs：URDF 中 <mesh filename="package://..."> 的检索根目录列表
        pkg = [] if package_dirs is None else package_dirs

        if use_collision:
            # 从 URDF 构建碰撞几何模型
            # 注：也可以用 buildModelsFromUrdf 一次性返回 (model, geom_models)
            self.collision_model: pin.GeometryModel = pin.buildGeomFromUrdf(
                self.model, urdf_path, pin.GeometryType.COLLISION, pkg
            )
            self.collision_data = self.collision_model.createData()

            # 默认：激活所有碰撞对；如需过滤，可调用 set_allowed_collision 关闭一部分
            # pinocchio 的 collision_model.collisionPairs 已经列出所有 pairs

        if use_visual:
            self.visual_model: pin.GeometryModel = pin.buildGeomFromUrdf(
                self.model, urdf_path, pin.GeometryType.VISUAL, pkg
            )
            self.visual_data = self.visual_model.createData()

        # 允许碰撞集（名字对），用来屏蔽相邻/同链等不关心的接触
        self.allowed_pairs = set()

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def link_names(self) -> List[str]:
        return [f.name for f in self.model.frames]

    @property
    def joint_limits(self):
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if name not in self.link_names:
            raise ValueError(
                f"{name} is not a link name. Valid link names: \n{self.link_names}"
            )
        return self.model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        joint_id = self.model.getFrameId(joint_name)
        parent_id = self.model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")
        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics & Geometry
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        J6 = pin.computeFrameJacobian(self.model, self.data, qpos, link_id, pin.ReferenceFrame.LOCAL)
        return J6

    def update_kinematics_and_geometry(self, qpos: npt.NDArray):
        """FK + 更新几何体位姿（碰撞/可视）"""
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)
        if self.collision_model is not None:
            pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data)
        if self.visual_model is not None:
            pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)

    # -------------------------------------------------------------------------- #
    # Collision API
    # -------------------------------------------------------------------------- #
    def set_allowed_collision(self, pairs: List[Tuple[str, str]]):
        """传入 (geom1_name, geom2_name) 的列表以‘允许’碰撞（即忽略这些对）"""
        self.allowed_pairs = set(tuple(sorted(p)) for p in pairs)

    def check_self_collision(self, threshold: float = 0.0):
        """
        返回发生碰撞（或距离 < threshold）的几何对列表：
        [(name1, name2, distance), ...]
        """
        if self.collision_model is None:
            raise RuntimeError("Collision model not loaded. Init with use_collision=True")

        results = []

        # True/False = 是否遇到碰撞就提前停止
        pin.computeCollisions(self.model, self.data,
                              self.collision_model, self.collision_data,
                              False)  # 遍历所有对

        # 逐对检查距离（更细粒度：得到最小距离）
        pin.computeDistances(self.model, self.data, self.collision_model, self.collision_data)

        for k, pair in enumerate(self.collision_model.collisionPairs):
            o1 = self.collision_model.geometryObjects[pair.first]
            o2 = self.collision_model.geometryObjects[pair.second]
            key = tuple(sorted((o1.name, o2.name)))
            if key in self.allowed_pairs:
                continue

            # 取该对的最近距离
            d = self.collision_data.distanceResults[k].min_distance
            # 有的版本字段名是 .min_distance 或 .minDistance，按你的 pin 版本调一下
            if d < threshold:
                results.append((o1.name, o2.name, float(d)))

        return results

    def min_distance(self):
        """
        返回 (dmin, (name1, name2))，方便做全局软约束
        """
        if self.collision_model is None:
            raise RuntimeError("Collision model not loaded. Init with use_collision=True")

        pin.computeDistances(self.model, self.data, self.collision_model, self.collision_data)
        dmin = float("+inf")
        pair_names = ("", "")
        for k, pair in enumerate(self.collision_model.collisionPairs):
            o1 = self.collision_model.geometryObjects[pair.first]
            o2 = self.collision_model.geometryObjects[pair.second]
            d = self.collision_data.distanceResults[k].min_distance
            if d < dmin:
                dmin = float(d)
                pair_names = (o1.name, o2.name)
        return dmin, pair_names
