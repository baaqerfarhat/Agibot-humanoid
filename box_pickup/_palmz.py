import numpy as np, mujoco, sys
from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg
m=get_x2_robot_cfg().spec_fn().compile(); d=mujoco.MjData(m)
BOX=np.array([0.4712,0.4587,0.4079]); OFF=np.array([0.0015,-0.0007,0.0058]); half=BOX/2
PALM=np.array([0.01,0,-0.10])
bid=lambda n: mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
HA=[bid("left_wrist_roll_link"),bid("right_wrist_roll_link")]
c=np.load(sys.argv[1],allow_pickle=True); cj=[str(x) for x in c["joint_names"]]
jp=np.asarray(c["joint_pos"],float); obj=np.asarray(c["object_pos_w"],float); oq=np.asarray(c["object_quat_w"],float)
mj=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
hinge=[j for j in mj if j in cj]
qadr=np.array([m.jnt_qposadr[mj.index(j)] for j in hinge])
q=np.zeros((len(jp),m.nq)); q[:,0:7]=jp[:,0:7]; q[:,qadr]=jp[:,7:][:,[cj.index(j) for j in hinge]]
z=obj[:,2]; off=np.flatnonzero(z>z.min()+0.010); g0,g1=off[0],off[-1]
print(f"grip window {g0}..{g1}   box half-height {half[2]:.3f} m")
print(f"{'frame':>6}{'boxCz':>8}{'L_localZ':>10}{'R_localZ':>10}{'  where on box'}")
for t in list(range(g0-30,g0+40,10))+list(range((g0+g1)//2,(g0+g1)//2+1)):
    d.qpos[:]=q[t]; mujoco.mj_forward(m,d)
    Rb=np.zeros(9); mujoco.mju_quat2Mat(Rb,oq[t]); Rb=Rb.reshape(3,3); cen=obj[t]+Rb@OFF
    lz=[]
    for h in HA:
        p=d.xpos[h]+d.xmat[h].reshape(3,3)@PALM
        lz.append((Rb.T@(p-cen))[2])
    tag=lambda v: "ABOVE TOP" if v>half[2] else ("upper" if v>0.05 else ("middle" if v>-0.05 else "lower"))
    print(f"{t:>6}{cen[2]:>8.3f}{lz[0]:>10.3f}{lz[1]:>10.3f}   L={tag(lz[0]):<9} R={tag(lz[1])}")
