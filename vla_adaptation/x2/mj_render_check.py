"""Can MuJoCo render X2 offscreen on this driver? Isaac's RTX renderer cannot."""
import os, sys, numpy as np, mujoco
CANDIDATES = [
 "/home/mtaheri/ws_AgibotX2/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml",
 "/home/mtaheri/ws_AgibotX2/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/x2/x2_31dof.xml",
 "/home/mtaheri/ws_AgibotX2/holosoma/src/holosoma/holosoma/data/robots/x2/x2_31dof.xml",
]
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
for path in CANDIDATES:
    if not os.path.exists(path):
        print(f"missing: {path}"); continue
    try:
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as e:
        print(f"LOAD FAIL {os.path.basename(path)}: {type(e).__name__}: {str(e)[:90]}"); continue
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"\nloaded {os.path.basename(path)}: nq={m.nq} nu={m.nu} nbody={m.nbody} "
          f"ngeom={m.ngeom} nmesh={m.nmesh}")
    try:
        with mujoco.Renderer(m, height=224, width=224) as r:
            r.update_scene(d)
            img = r.render()
        img = np.asarray(img)
        print(f"  RENDER OK  {img.shape} {img.dtype}  min {img.min()} max {img.max()} "
              f"mean {img.mean():.1f}  {'NONBLACK' if img.max() > 0 else 'ALL BLACK'}")
        import imageio.v2 as iio
        out = f"/tmp/claude-1021/-home-mtaheri-ws-AgibotX2/b956f795-a85b-442d-9001-98cf4a7b3626/scratchpad/mj_{os.path.basename(path).replace('.xml','')}.png"
        iio.imwrite(out, img); print(f"  wrote {out}")
    except Exception as e:
        print(f"  RENDER FAIL: {type(e).__name__}: {str(e)[:110]}")
