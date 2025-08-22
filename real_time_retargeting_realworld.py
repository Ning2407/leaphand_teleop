import time
import multiprocessing
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from queue import Empty
import tyro
from loguru import logger
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.single_hand_detector import SingleHandDetector


from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)

# ---- 你提供的 Dynamixel 客户端 ----
from dex_retargeting.utils.dynamixel_client import DynamixelClient

# 简单EMA抑制抖动（可选）
class EMA:
    def __init__(self, alpha=0.25):
        self.a = alpha; self.x = None
    def __call__(self, v):
        if self.x is None: self.x = v
        self.x = self.a * v + (1 - self.a) * self.x
        return self.x

def start_retargeting(queue, robot_dir: str, config_path: str):
    # 1) 构建 retargeting
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # 2) 驱动侧关节名顺序 & 舵机ID（必须按顺序一一对应）
    driver_joint_names = [
        "0",   # index finger pip
        "1",   # index finger mcp
        "2",   # index finger dip
        "3",   # index fingertip
        "4",   # middle finger pip
        "5",   # middle finger mcp
        "6",   # middle finger dip
        "7",   # middle fingertip
        "8",   # ring finger pip
        "9",   # ring finger mcp
        "10",  # ring finger dip
        "11",  # ring fingertip
        "12",  # thumb base
        "13",  # thumb pip
        "14",  # thumb dip
        "15",  # thumb fingertip
    ]
    motor_ids = [
        # ====== 与上面顺序一一对应的 Dynamixel ID ======

        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    ]

    # 3) retargeting 输出顺序 → 驱动顺序 的索引映射
    retarget_names = retargeting.joint_names
    idx_retarget_to_driver = np.array(
        [retarget_names.index(n) for n in driver_joint_names], dtype=int
    )

    # 4) 取限位，用于裁剪 —— 兼容不同实现
    n = len(driver_joint_names)
    lower = upper = None

    # 优先尝试：retargeting.robot_wrapper.joint_limits（很多项目这样放）
    if hasattr(retargeting, "robot_wrapper") and hasattr(retargeting.robot_wrapper, "joint_limits"):
        jl = retargeting.robot_wrapper.joint_limits  # shape (nq, 2)
        lower, upper = jl[:, 0], jl[:, 1]

    # 备选：optimizer.robot / optimizer.model 等位置
    elif hasattr(retargeting.optimizer, "robot") and hasattr(retargeting.optimizer.robot, "joint_limits"):
        jl = retargeting.optimizer.robot.joint_limits
        lower, upper = jl[:, 0], jl[:, 1]

    # 实在没有：先不裁剪，后面 np.clip 将被跳过
    if lower is None or upper is None:
        lower_drv = None
        upper_drv = None
    else:
        lower_drv = lower[idx_retarget_to_driver]
        upper_drv = upper[idx_retarget_to_driver]


    # 5) 初始化 Dynamixel 通信
    #    注意：pos_scale 默认按 X 系列 0~4096 ticks = 2π rad；如你的电机不是该规格，请改 DEFAULT_POS_SCALE
    dxl = DynamixelClient(motor_ids=motor_ids, port="/dev/ttyUSB0", baudrate=1000000)
    dxl.connect()

    # (可选强烈推荐) 切换“位置模式”，X系列通常 3=位置/4=扩展位置/16=电流限制位置 等；按你的电机手册填写
    POSITION_MODE = 3
    dxl.set_operation_mode(motor_ids, POSITION_MODE)

    # 开启力矩
    dxl.set_torque_enabled(motor_ids, True)

    # 6) 平滑/频率/看门狗
    ema = EMA(alpha=0.25)     # 抖动明显可略调大
    rate_hz = 60              # 目标发送频率
    min_dt = 1.0 / rate_hz
    watchdog_timeout = 0.5    # 0.5s 未检测到手 → 停住
    last_send = 0.0
    last_seen = time.time()

    try:
        while True:
            # 读取相机帧
            try:
                bgr = queue.get(timeout=5)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Empty:
                logger.error("5秒内摄像头无数据，退出。")
                break

            # 手部检测
            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            cv2.imshow("leap_hand_teleop", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            now = time.time()

            # 看门狗：手丢失时保持or失能
            if joint_pos is None:
                if now - last_seen > watchdog_timeout:
                    # “停住”建议：写当前反馈位置或安全位，也可以直接禁用力矩（更保险）
                    # pos_fb = dxl.read_pos()   # 读当前关节（弧度）
                    # dxl.write_desired_pos(motor_ids, pos_fb)
                    pass
                continue
            else:
                last_seen = now

            # 组装 retarget 输入
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting.optimizer.retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                o_idx, t_idx = indices[0, :], indices[1, :]
                ref_value = joint_pos[t_idx, :] - joint_pos[o_idx, :]

            # 求解 qpos（弧度）
            qpos = retargeting.retarget(ref_value)

            # 映射到驱动顺序 & 限位裁剪
            qpos_drv = qpos[idx_retarget_to_driver]
            if lower_drv is not None and upper_drv is not None:
                qpos_drv = np.clip(qpos_drv, lower_drv, upper_drv)

            # （可选）自碰投影：qpos_drv = project_self_collision(qpos_drv)
            # （可选）EMA 平滑
            qpos_drv = ema(qpos_drv)
            qpos_drv = np.pi + qpos_drv
            # 频率限速发送
            if now - last_send >= min_dt:
                dxl.write_desired_pos(motor_ids, qpos_drv)  # 内部已按 pos_scale 转换成刻度
                print(qpos_drv)
                last_send = now

    finally:
        # 退出前关断力矩并断开
        try:
            dxl.set_torque_enabled(motor_ids, False, retries=0)
        except Exception:
            pass
        dxl.disconnect()
        cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import sapien
# import tyro
# from loguru import logger
# from sapien.asset import create_dome_envmap
# from sapien.utils import Viewer

# from dex_retargeting.retargeting_config import RetargetingConfig
# from dex_retargeting.single_hand_detector import SingleHandDetector


# def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
#     RetargetingConfig.set_default_urdf_dir(str(robot_dir))
#     logger.info(f"Start retargeting with config {config_path}")
#     retargeting = RetargetingConfig.load_from_file(config_path).build()

#     hand_type = "Right" if "right" in config_path.lower() else "Left"
#     detector = SingleHandDetector(hand_type=hand_type, selfie=False)

#     sapien.render.set_viewer_shader_dir("default")
#     sapien.render.set_camera_shader_dir("default")

#     config = RetargetingConfig.load_from_file(config_path)

#     # Setup
#     scene = sapien.Scene()
#     render_mat = sapien.render.RenderMaterial()
#     render_mat.base_color = [0.06, 0.08, 0.12, 1]
#     render_mat.metallic = 0.0
#     render_mat.roughness = 0.9
#     render_mat.specular = 0.8
#     scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

#     # Lighting
#     scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
#     scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
#     scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
#     scene.set_environment_map(
#         create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
#     )
#     scene.add_area_light_for_ray_tracing(
#         sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
#     )

#     # Camera
#     cam = scene.add_camera(
#         name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
#     )
#     cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

#     viewer = Viewer()
#     viewer.set_scene(scene)
#     viewer.control_window.show_origin_frame = False
#     viewer.control_window.move_speed = 0.01
#     viewer.control_window.toggle_camera_lines(False)
#     viewer.set_camera_pose(cam.get_local_pose())

#     # Load robot and set it to a good pose to take picture
#     loader = scene.create_urdf_loader()
#     filepath = Path(config.urdf_path)
#     robot_name = filepath.stem
#     loader.load_multiple_collisions_from_file = True
#     if "ability" in robot_name:
#         loader.scale = 1.5
#     elif "dclaw" in robot_name:
#         loader.scale = 1.25
#     elif "allegro" in robot_name:
#         loader.scale = 1.4
#     elif "shadow" in robot_name:
#         loader.scale = 0.9
#     elif "bhand" in robot_name:
#         loader.scale = 1.5
#     elif "leap" in robot_name:
#         loader.scale = 1.4
#     elif "svh" in robot_name:
#         loader.scale = 1.5

#     if "glb" not in robot_name:
#         filepath = str(filepath).replace(".urdf", "_glb.urdf")
#     else:
#         filepath = str(filepath)

#     robot = loader.load(filepath)

#     if "ability" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.15]))
#     elif "shadow" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.2]))
#     elif "dclaw" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.15]))
#     elif "allegro" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.05]))
#     elif "bhand" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.2]))
#     elif "leap" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.15]))
#     elif "svh" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.13]))

#     # Different robot loader may have different orders for joints
#     sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
#     retargeting_joint_names = retargeting.joint_names
#     retargeting_to_sapien = np.array(
#         [retargeting_joint_names.index(name) for name in sapien_joint_names]
#     ).astype(int)

#     while True:
#         try:
#             bgr = queue.get(timeout=5)
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#         except Empty:
#             logger.error(
#                 "Fail to fetch image from camera in 5 secs. Please check your web camera device."
#             )
#             return

#         _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
#         bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
#         cv2.imshow("realtime_retargeting_demo", bgr)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#         if joint_pos is None:
#             logger.warning(f"{hand_type} hand is not detected.")
#         else:
#             retargeting_type = retargeting.optimizer.retargeting_type
#             indices = retargeting.optimizer.target_link_human_indices
#             if retargeting_type == "POSITION":
#                 indices = indices
#                 ref_value = joint_pos[indices, :]
#             else:
#                 origin_indices = indices[0, :]
#                 task_indices = indices[1, :]
#                 ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
#             qpos = retargeting.retarget(ref_value)


#             robot.set_qpos(qpos[retargeting_to_sapien])
#             print(qpos[retargeting_to_sapien])

#         for _ in range(2):
#             viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=2)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
