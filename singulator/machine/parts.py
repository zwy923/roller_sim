"""Dimensions and MJCF snippets shared by the sections of the machine.

Every conveyor here is a plate whose position is held and whose velocity is prescribed (physics/drives.py), which
for contact is exactly a belt surface. Equipment never collides with equipment in the simulation (contype 2 /
conaffinity 0): lumps touch equipment, equipment passes through equipment.
"""
BELT_THICKNESS = .04            # m: the plate that stands for a belt
SKIRT = dict(height=.50, thickness=.02)
LUMP_TOP = .50                  # m: a lump reaches this high above the surface it lies on (the tallest it can lie)


def f3(values):
    return ' '.join('%.10g' % v for v in values)


def plate_belt(name, x0, x1, y, top_z, half_w, rgba):
    """A belt from x0 to x1, its top at top_z: body `name` on the slide joint `name`j."""
    return ('<body name="%s" pos="%.4f %.4f %.4f"><joint name="%sj" type="slide" axis="1 0 0" armature="10000"/>'
            '<inertial pos="0 0 0" mass="5" diaginertia=".1 .1 .1"/>'
            '<geom class="belt" type="box" size="%.4f %.4f %.4f" rgba="%s"/></body>'
            % (name, (x0 + x1) / 2, y, top_z - BELT_THICKNESS / 2, name,
               (x1 - x0) / 2, half_w, BELT_THICKNESS / 2, rgba))


def pulley(name, dia, cx, y, cz, half_w, rgba):
    """A drum or pulley across the belt, turning with it (surface speed = belt speed): body `name`, hinge `name`j."""
    return ('<body name="%s" pos="%.4f %.4f %.4f"><joint name="%sj" type="hinge" axis="0 1 0" armature="1000"/>'
            '<inertial pos="0 0 0" mass="20" diaginertia=".2 .2 .2"/>'
            '<geom class="belt" type="cylinder" fromto="0 %.4f 0 0 %.4f 0" size="%.4f" rgba="%s"/></body>'
            % (name, cx, y, cz, name, -half_w, half_w, dia / 2, rgba))


def skirt(name, y, xa, xb, z0):
    """A flat skirt plate along x from xa to xb at y, standing on z0."""
    return ('<geom name="%s" class="wall" type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f %.4f" rgba=".55 .55 .6 .45"/>'
            % (name, (xb - xa) / 2, SKIRT['thickness'] / 2, SKIRT['height'] / 2,
               (xa + xb) / 2, y, z0 + SKIRT['height'] / 2))
