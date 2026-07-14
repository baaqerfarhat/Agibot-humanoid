"""Generate the X2 retargeting MJCF from the mjlab X2 model.

The interaction-mesh retargeter needs:
  * a named, collidable ground plane geom ("ground"),
  * named, collidable robot collision geoms (contype/conaffinity not both 0),
  * foot-contact sphere BODIES under each ankle_roll link for the foot-sticking
    constraint (looked up by body name via forward kinematics),
  * a floating base (freejoint).

This script parses the mjlab MJCF and rewrites it into
``models/x2/x2_31dof.xml`` (robot-only / ground) so the result stays
reproducible. Run from the holosoma_retargeting package root.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("/home/baaqer/baaqer_ws/mjlab/src/mjlab/asset_zoo/robots/x2/xmls/x2.xml")
OUT = Path(__file__).parent / "x2_31dof.xml"

# Foot contact spheres in the ankle_roll frame (mirror G1's 4-sphere layout),
# taken from the mjlab foot collision geom positions.
FOOT_SPHERES = {
    "1": (-0.05, 0.05, -0.068),   # heel, inner
    "2": (-0.05, -0.05, -0.068),  # heel, outer
    "3": (0.11, 0.05, -0.068),    # toe, inner
    "4": (0.11, -0.05, -0.068),   # toe, outer
    "5": (0.139, 0.0, -0.066),    # toe, front-center (toe keypoint)
}


def main() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()

    # 1) compiler: meshes live in ./assets
    comp = root.find("compiler")
    comp.set("meshdir", "assets")

    # 2) make the collision default class actually collidable
    for d in root.iter("default"):
        if d.get("class") == "collision":
            g = d.find("geom")
            if g is not None:
                g.set("contype", "1")
                g.set("conaffinity", "1")

    # 3) name every collision geom (retargeter indexes geoms by name)
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("class") == "collision" and geom.get("name") is None:
                mesh = geom.get("mesh")
                base = body.get("name")
                geom.set("name", f"{base}_collision" if mesh is None else mesh)

    # 4) drop sensors and imu sites/geoms we do not need for retargeting
    for tag in ("sensor",):
        el = root.find(tag)
        if el is not None:
            root.remove(el)
    for body in root.iter("body"):
        for site in body.findall("site"):
            body.remove(site)

    # 5) add foot-sticking sphere bodies under each ankle_roll link
    for body in root.iter("body"):
        name = body.get("name")
        if name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            side = "left" if name.startswith("left") else "right"
            for idx, pos in FOOT_SPHERES.items():
                sb = ET.SubElement(body, "body")
                sb.set("name", f"{side}_ankle_roll_sphere_{idx}_link")
                sb.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
                inert = ET.SubElement(sb, "inertial")
                inert.set("pos", "0 0 0")
                inert.set("mass", "0.001")
                inert.set("diaginertia", "1e-7 1e-7 1e-7")
                vg = ET.SubElement(sb, "geom")
                vg.set("type", "sphere")
                vg.set("size", "0.005")
                vg.set("contype", "0")
                vg.set("conaffinity", "0")
                vg.set("group", "1")
                vg.set("rgba", "1 0.5 0 1")
                cg = ET.SubElement(sb, "geom")
                cg.set("name", f"{side}_ankle_roll_sphere_{idx}_link")
                cg.set("type", "sphere")
                cg.set("size", "0.005")

    # 5b) add a hand-contact marker body at the palm/hand center under each
    # wrist_roll link. The arm marker for the interaction mesh is looked up by
    # BODY name via FK, and the SMPL-H wrist maps to this body. The wrist_roll
    # frame origin sits at the wrist JOINT, but the hand geometry center is
    # ~0.13m further along the forearm (local -z). Placing the marker there pulls
    # the retargeted box down to the robot's actual grasping surface so the hands
    # rest on the box instead of hovering above it.
    HAND_CONTACT_OFFSET = (0.02, 0.0, -0.13)  # (x fwd, y, z toward hand tip)
    for body in root.iter("body"):
        name = body.get("name")
        if name in ("left_wrist_roll_link", "right_wrist_roll_link"):
            side = "left" if name.startswith("left") else "right"
            hb = ET.SubElement(body, "body")
            hb.set("name", f"{side}_hand_contact_link")
            hb.set("pos", f"{HAND_CONTACT_OFFSET[0]} {HAND_CONTACT_OFFSET[1]} {HAND_CONTACT_OFFSET[2]}")
            inert = ET.SubElement(hb, "inertial")
            inert.set("pos", "0 0 0")
            inert.set("mass", "0.001")
            inert.set("diaginertia", "1e-7 1e-7 1e-7")
            # Pure tracking marker for the interaction mesh (looked up by body
            # name via FK). It must NOT be collidable: a collidable hand sphere
            # would add a non-penetration constraint that directly opposes the
            # interaction-mesh cost pulling the box to the hand, which
            # over-constrains the QP (infeasible) and re-opens the contact gap.
            vg = ET.SubElement(hb, "geom")
            vg.set("type", "sphere")
            vg.set("size", "0.02")
            vg.set("contype", "0")
            vg.set("conaffinity", "0")
            vg.set("group", "1")
            vg.set("rgba", "0 1 0 1")

    # 6) add ground plane + light + plane material to the worldbody / assets
    worldbody = root.find("worldbody")
    ground = ET.Element("geom")
    ground.set("name", "ground")
    ground.set("type", "plane")
    ground.set("size", "10 10 0.1")
    ground.set("pos", "0 0 0")
    ground.set("material", "MatPlane")
    ground.set("contype", "1")
    ground.set("conaffinity", "1")
    worldbody.insert(0, ground)
    light = ET.Element("light")
    light.set("pos", "0 0 1000")
    light.set("castshadow", "true")
    worldbody.insert(1, light)

    asset = ET.SubElement(root, "asset")
    tex = ET.SubElement(asset, "texture")
    tex.set("name", "texplane")
    tex.set("builtin", "checker")
    tex.set("height", "512")
    tex.set("width", "512")
    tex.set("rgb1", ".2 .3 .4")
    tex.set("rgb2", ".1 .15 .2")
    tex.set("type", "2d")
    mat = ET.SubElement(asset, "material")
    mat.set("name", "MatPlane")
    mat.set("reflectance", "0.5")
    mat.set("shininess", "0.01")
    mat.set("specular", "0.1")
    mat.set("texrepeat", "1 1")
    mat.set("texture", "texplane")
    mat.set("texuniform", "true")

    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="unicode", xml_declaration=False)
    print(f"wrote {OUT}")

    # --- object_interaction variant: robot + free-joint largebox --------------
    # Mirror models/g1/g1_29dof_w_largebox.xml: add the largebox mesh asset and a
    # free-floating "largebox_link" body with a collidable geom named "largebox"
    # (the retargeter looks up the object by this geom/body name).
    box_mesh = ET.SubElement(asset, "mesh")
    box_mesh.set("name", "largebox_mesh")
    box_mesh.set("file", "../../largebox/largebox.obj")
    box_mesh.set("scale", "1 1 1")

    box_body = ET.SubElement(worldbody, "body")
    box_body.set("name", "largebox_link")
    ET.SubElement(box_body, "freejoint")
    box_inert = ET.SubElement(box_body, "inertial")
    box_inert.set("pos", "0 0 0")
    box_inert.set("mass", "0.1")
    box_inert.set("diaginertia", "0.002 0.002 0.002")
    box_geom = ET.SubElement(box_body, "geom")
    box_geom.set("name", "largebox")
    box_geom.set("type", "mesh")
    box_geom.set("mesh", "largebox_mesh")
    box_geom.set("contype", "1")
    box_geom.set("conaffinity", "1")
    box_geom.set("pos", "0 0 0")
    box_geom.set("quat", "1 0 0 0")
    box_geom.set("rgba", "0.7 0.8 0.9 0.7")
    box_geom.set("friction", "0.9 0.5 0.5")
    box_geom.set("solref", "0.02 1")
    box_geom.set("solimp", "0.9 0.95 0.001")

    out_box = OUT.parent / "x2_31dof_w_largebox.xml"
    ET.indent(tree, space="  ")
    tree.write(out_box, encoding="unicode", xml_declaration=False)
    print(f"wrote {out_box}")


if __name__ == "__main__":
    main()
