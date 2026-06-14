"""
This script collects reference trajectory data under different impedance control gains.
It moves the Franka robot arm in a sinusoidal pattern while logging joint positions, velocities,
desired positions, and control torques. The collected data is saved to a .npz file for further analysis.
"""

import asyncio
import numpy as np
from aiofranka.robot import RobotInterface
from aiofranka import FrankaController
import time
import os
import matplotlib.pyplot as plt
from pylibfranka import Gripper


async def main():
    robot = RobotInterface("172.16.0.2")
    controller = FrankaController(robot)
    await controller.start()

    # Test with your target control gains
    # kp = 128
    # kd = 8
    # base = np.array([1, 1, 1, 1, 0.1, 0.1, 0.05])

    kp = 250
    kd = 26
    base = np.array([1, 1, 1, 1, 0.6, 0.6, 0.6])

    # Move to initial position
    with controller.state_lock:
        controller.kp = base * 80
        controller.kd = base * 4
    await controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])
    await asyncio.sleep(2.0)

    # Start impedance control with target gains
    controller.switch("impedance")
    freq = 20
    controller.set_freq(freq)  # Higher frequency for better data
    with controller.state_lock:
        controller.kp = base * kp
        controller.kd = base * kd
        print("Kp: ", controller.kp)
        print("Kd: ", controller.kd)

    logs = {"qpos": [], "qvel": [], "qdes": [], "ctrl": [], "time": []}

    # Multi-frequency excitation per joint
    duration = 15.0  # seconds
    n_steps = int(duration * freq)

    for cnt in range(n_steps*100):
        t = cnt / freq
        logs["time"].append(t)
        logs["qpos"].append(controller.robot.data.qpos.copy())
        logs["qvel"].append(controller.robot.data.qvel.copy())
        logs["ctrl"].append(controller.robot.data.ctrl.copy())
        logs["qdes"].append(controller.q_desired.copy())

        
        init = controller.initial_qpos
        delta = 0.5 * np.array(
            [
                0.2 * np.sin(2 * np.pi * 0.3 * t),  
                0.3 * np.sin(2 * np.pi * 0.5 * t),   
                0.15 * np.sin(2 * np.pi * 0.4 * t),  
                0.15 * np.sin(2 * np.pi * 0.6 * t),  
                0.2 * np.sin(2 * np.pi * 0.7 * t),  
                0.2 * np.sin(2 * np.pi * 0.8 * t),  
                0.2 * np.sin(2 * np.pi * 0.9 * t),  
            ]
        )

        await controller.set("q_desired", init + delta)

    await asyncio.sleep(1.0)

    for key in logs.keys():
        logs[key] = np.stack(logs[key])

    os.makedirs("./sysid_data/", exist_ok=True)
    np.savez(f"./sysid_data/franka_system_identification_260224_{int(kp)}_Kd{int(kd)}.npz", **logs)


if __name__ == "__main__":
    asyncio.run(main())
