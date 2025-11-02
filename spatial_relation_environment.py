import numpy as np

def local_transformation(ag_pos, ag_rot, obj_pos):
    assert len(ag_rot) == 3
    v = np.array(obj_pos) - np.array(ag_pos)
    # +X = right, -X = left
    # +Y = up
    # +Z = forward, -Z = backward
    vx, vy, vz = v
    # yaw = np.deg2rad(ag_rot[1])
    yaw = np.deg2rad(ag_rot[1])
    # apply rotation matrix for transforming a world vector into the agent's local coordinates
    x_local = (np.cos(yaw) * vx) - (np.sin(yaw) * vz)
    z_local = (np.sin(yaw) * vx) + (np.cos(yaw) * vz)
    theta = np.degrees(np.arctan2(x_local, z_local))
    # return (x, y, z), yaw, theta
    return x_local, z_local, theta

def relative_direction(angle):
    # Decide qualitative direction
    if -22.5 <= angle <= 22.5:
        return "front"
    elif 22.5 < angle <= 67.5:
        return "front-right"
    elif 67.5 < angle <= 112.5:
        return "right"
    elif -67.5 >= angle > -112.5:
        return "left"
    elif -157.5 < angle < -112.5:
        return "back-left"
    elif 157.5 > angle > 112.5:
        return "back-right"
    elif angle >= 157.5 or angle <= -157.5:
        return "back"
    else:
        return "unknown"

if __name__ == "__main__":
    ag_pos = np.array((10.75, 0.9, 0.85))
    # ag_pos = (11.05, 0.9, 1.52)
    ag_rot = np.array((0, 0, 0))
    # obj_pos = np.array((10.55, 0, 1.85))
    obj_pos = (9.86, 0.85, 2.17)
    x_l, z_l, theta = local_transformation(ag_pos, ag_rot, obj_pos)
    print(x_l, z_l, theta)
    dir = relative_direction(theta)
    print(dir)