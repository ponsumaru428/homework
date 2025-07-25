import pybullet as p
import pybullet_data
import numpy as np
import time

# 曲線生成（サインカーブ）
def generate_curve(num_points=500, length=10.0):
    x = np.linspace(0, length, num_points)
    y = np.sin(2 * np.pi * x / length)  # サインカーブ
    z = np.zeros_like(x)
    return np.vstack((x, y, z)).T

# 曲線描画
def draw_curve(points, color=[1, 0, 0], width=2):
    for i in range(len(points) - 1):
        p.addUserDebugLine(points[i], points[i + 1], lineColorRGB=color, lineWidth=width)

# Pure Pursuit 的な目標追従
def pure_pursuit_controller(car_pos, curve_points, lookahead=0.5):
    dists = np.linalg.norm(curve_points - car_pos, axis=1)
    nearest_index = np.argmin(dists)
    for i in range(nearest_index, len(curve_points)):
        if np.linalg.norm(curve_points[i] - car_pos) > lookahead:
            return curve_points[i]
    return curve_points[-1]

# メイン実行
if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")
    curve = generate_curve()
    draw_curve(curve)

    car = p.loadURDF("racecar/racecar.urdf", basePosition=curve[0].tolist())

    for step in range(3000):
        pos, _ = p.getBasePositionAndOrientation(car)
        pos = np.array(pos)

        target = pure_pursuit_controller(pos, curve, lookahead=0.5)
        direction = target - pos
        direction[2] = 0
        norm = np.linalg.norm(direction)

        # ゆっくり移動（速度係数 0.2）
        velocity = (direction / norm * 0.2).tolist() if norm > 1e-3 else [0, 0, 0]

        p.resetBaseVelocity(car, linearVelocity=velocity)
        p.stepSimulation()

        # 表示もゆっくり
        time.sleep(1./120.)

    p.disconnect()

