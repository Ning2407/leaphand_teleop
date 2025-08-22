
import time
import math
import numpy as np
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dex_retargeting.utils.dynamixel_client import DynamixelClient

class LeapHand():
    def __init__(self, hand_name):

        self.motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.hand_name = hand_name

        if(hand_name == "left"):
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 1000000)

        elif(hand_name == "right"):
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 1000000)

        connected = False
        while not connected:
            try:
                self.dxl_client.connect()
                connected = True
            except OSError as e:
                print(f"Serial open failed, retrying in 1s…")
                time.sleep(1)

        # 初始化pid参数
        self.kP = 800.0
        self.kI = 0.0
        self.kD = 0.0
        self.curr_lim = 350.0
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)  # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)  # Igain
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)  # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)  # Dgain damping for side to side should be a bit less
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)

        self.hand_actual_pose = None
        self.desired_pose = None
        print('init finished')


    def deg_to_rad(self, deg):
        return deg * math.pi / 180.0



    def send_pose(self):
        step_size = 0.2
        threshold = self.deg_to_rad(10)
        self.hand_actual_pose = np.array(self.dxl_client.read_pos())
        if self.desired_pose is not None: 
            # We do pose differentiation and move the hand
            target = self.desired_pose
            curr_pos = self.hand_actual_pose
            delta = target - curr_pos
            command = curr_pos
            for i in range(len(delta)):
                if np.abs(delta[i]) < threshold:
                    command[i] = target[i]
                else:
                    if delta[i] > 0:
                        command[i] = curr_pos[i] + step_size
                    else:
                        command[i] = curr_pos[i] - step_size

            self.dxl_client.write_desired_pos(self.motors, command) 


# if __name__ == '__main__':
#     left_hand = LeapHand('left')
#     # left_hand.pose_right_one()
#     left_hand.pose_right_cup_fix()
#     # left_hand.pose_right_cup_relax()