import numpy as np, mujoco
from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg

BOX = np.array([0.4712, 0.4587, 0.4079])
OFF = np.array([0.0015, -0.0007, 0.0058])
PALM = np.array([0.01, 0, -0.10])
half = BOX / 2

m = get_x2_robot_cfg().spec_fn().compile()
d = mujoco.MjData(m)
c = np.load(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_minimal.npz",
    allow_pickle=True,
)
cj = [str(x) for x in c["joint_names"]]
jp = np.asarray(c["joint_pos"], float)
obj = np.asarray(c["object_pos_w"], float)
oq = np.asarray(c["object_quat_w"], float)
mj = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
hinge = [j for j in mj if j in cj]
qadr = np.array([m.jnt_qposadr[mj.index(j)] for j in hinge])
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
HA = [bid("left_wrist_roll_link"), bid("right_wrist_roll_link")]
skip = set()
for b in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if any(k in name for k in ("wrist", "palm", "hand")):
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            skip.add(g)
coll = [
    g
    for g in range(m.ngeom)
    if (m.geom_contype[g] or m.geom_conaffinity[g]) and g not in skip
]


def sdf(p, cen, Rb, r):
    dd = np.abs(Rb.T @ (p - cen)) - half
    return float(np.linalg.norm(np.maximum(dd, 0)) + min(dd.max(), 0) - r)


print(f"{'fr':>5}{'pelv':>7}{'Lxyz':>24}{'Rxyz':>24}{'legmm':>8}  who")
for t in (0, 50, 80, 100, 110, 140, 200):
    d.qpos[:] = 0
    d.qpos[0:7] = jp[t, 0:7]
    d.qpos[qadr] = jp[t, 7:][[cj.index(j) for j in hinge]]
    mujoco.mj_forward(m, d)
    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, oq[t])
    Rb = Rb.reshape(3, 3)
    cen = obj[t] + Rb @ OFF
    locs = []
    for h in HA:
        p = d.xpos[h] + d.xmat[h].reshape(3, 3) @ PALM
        locs.append(Rb.T @ (p - cen))
    worst, who = 9e9, ""
    for g in coll:
        gap = sdf(d.geom_xpos[g], cen, Rb, m.geom_rbound[g])
        if gap < worst:
            worst = gap
            who = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
    print(
        f"{t:5d}{d.xpos[bid('pelvis')][2]:7.3f}"
        f" L[{locs[0][0]:+.2f},{locs[0][1]:+.2f},{locs[0][2]:+.2f}]"
        f" R[{locs[1][0]:+.2f},{locs[1][1]:+.2f},{locs[1][2]:+.2f}]"
        f"{worst * 1e3:8.0f}  {who}"
    )
print("side-face center is local z=0, y=0; lid is z=+0.20")
