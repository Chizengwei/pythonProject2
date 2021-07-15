# -*- coding:utf-8 -*-

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Created Date: Tuesday, February 5th 2019, 10:18:04 am
# Copyright: Tommi Hyppänen

# format: black, lint: flake8, 120 horizontal ruler

import numpy as np
import math
import bmesh
import bpy
import time
import mathutils
import gpu
from gpu_extras.batch import batch_for_shader
import bgl
from collections import defaultdict
import sys
import site

bpy_intersect = mathutils.geometry.intersect_line_line_2d


# import random

# import tracemalloc
# tracemalloc.start()

bl_info = {
    "name": "Shotpacker",
    "category": "UV",
    "description": "Pack UV islands to save space.",
    "author": "Tommi Hyppänen (ambi)",
    "location": "UV Editor > Sidebar > Shotpacker > Pack",
    "version": (2, 0, 18),
    "blender": (2, 80, 0),
}

MODULES_LOADED = True
PI = math.pi

# add user site to sys.path
# binaries go to {site.USER_BASE}/bin
# venv notice:
#   https://stackoverflow.com/questions/33412974/
#   how-to-uninstall-a-package-installed-with-pip-install-user/56948334#56948334
app_path = site.USER_SITE
if app_path not in sys.path:
    sys.path.append(app_path)

try:
    if "packer_iface" not in locals():
        from .bpy_amb import fastmesh as amb_fm
        from .bpy_amb import bbmesh as amb_bm
        from .bpy_amb import utils as amb_util
        from .bpy_amb import image as amb_image
        from . import island  # noqa: F401
        from . import uv
        from . import packer
        from . import cgeo

        # from . import accelerator
        from . import packer_iface  # noqa: F401
        from . import tools
    else:
        if "packer" not in locals():
            from . import packer
        import importlib

        importlib.reload(island)
        importlib.reload(uv)
        importlib.reload(packer)
        importlib.reload(cgeo)
        # importlib.reload(accelerator)
        importlib.reload(packer_iface)
        importlib.reload(tools)
        importlib.reload(amb_fm)
        importlib.reload(amb_bm)
        importlib.reload(amb_util)
        importlib.reload(amb_image)

except ModuleNotFoundError as mnfe:
    print("!!! Shotpacker exception: Required module not found !!!")
    print(
        "Please install all required modules from the Blender preferences"
        " -> Add-ons -> Shotpacker preferences"
    )
    print(mnfe)
    MODULES_LOADED = False


class ShotpackerInstallLibraries(bpy.types.Operator):
    bl_idname = "uv.spacker_install_libraries"
    bl_label = "Install required libraries: pip, cffi,  pyclipper"

    def execute(self, context):
        import subprocess
        import os
        import platform
        import sys

        if platform.system() == "Windows":
            print("Platform: Windows")
        elif platform.system() == "Darwin":
            print("Platform: Mac")
        elif platform.system() == "Linux":
            print("Platform: Linux")
        else:
            print("!!! Unknown system !!!")

        # site_path = ""
        # print(site_path)
        print(">>> Site packages:")
        for path in sys.path:
            if os.path.basename(path) in ("dist-packages", "site-packages"):
                print(path)

        # import site
        # print(site.getsitepackages())
        # from distutils.sysconfig import get_python_lib
        # print(get_python_lib())

        # import ensurepip
        # ensurepip.bootstrap()
        # os.environ.pop("PIP_REQ_TRACKER", None)

        pp = bpy.app.binary_path_python
        print(">>> Using", pp)

        subprocess.run([pp, "-m", "ensurepip", "--user"], check=True)

        # First uninstall everything with PIP because
        # PIP and Blender don't necessarily share the same python path for modules
        subprocess.run([pp, "-m", "pip", "uninstall", "-y", "cffi"], check=True)
        subprocess.run([pp, "-m", "pip", "uninstall", "-y", "pyclipper"], check=True)
        subprocess.run([pp, "-m", "pip", "uninstall", "-y", "pycparser"], check=True)

        print(">>> Installing Shotpacker libraries...")

        # On Windows doing pip install --user confuses Blender and it can't find the libs :(

        # First try to install into Blender folder
        subprocess.run([pp, "-m", "pip", "install", "--user", "cffi"], check=True)
        subprocess.run([pp, "-m", "pip", "install", "--user", "pyclipper"], check=True)

        try:
            import cffi
            import pyclipper
        except ModuleNotFoundError as mnfe:
            print(">>> User install didn't work, trying admin...")
            subprocess.call([pp, "-m", "pip", "install", "cffi"])
            subprocess.call([pp, "-m", "pip", "install", "pyclipper"])

        from . import packer  # noqa: F401
        from .bpy_amb import fastmesh as amb_fm
        from .bpy_amb import bbmesh as amb_bm
        from .bpy_amb import utils as amb_util
        from .bpy_amb import image as amb_image
        from . import island  # noqa: F401
        from . import uv
        from . import packer
        from . import cgeo

        # from . import accelerator
        from . import packer_iface  # noqa: F401
        from . import tools

        global MODULES_LOADED
        MODULES_LOADED = True

        return {"FINISHED"}


class ShotpackerUninstallAccelerator(bpy.types.Operator):
    bl_idname = "uv.spacker_uninstall"
    bl_label = "Unload acceleration module"
    bl_description = "Unload acceleration module. For disabling the addon before removing it."

    def execute(self, context):
        if "shotpacker.accelerator" in sys.modules:
            the_lib = sys.modules["shotpacker.accelerator"]
            the_lib.ffi.dlclose(the_lib.ffilib)
            print("Acceleration module disabled")
        else:
            print("No module found")
            # for i in sys.modules:
            #     if "packer" in i:
            #         print(i)
        return {"FINISHED"}


class ShotpackerAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        if MODULES_LOADED is False:
            row.operator(ShotpackerInstallLibraries.bl_idname, text="Install required libraries")
        else:
            # # row.label(text="All libraries loaded. Addon working.")
            # row = layout.row()
            # row.operator(
            #     ShotpackerUninstallAccelerator.bl_idname, text="Disable acceleration module"
            # )

            # row = layout.row()
            # row.prop(context.scene.s_packing, "use_pixel_margins",
            #   text="Use pixel margin setting")
            # if bpy.context.scene.s_packing.use_pixel_margins:
            row = layout.row()
            row.prop(context.scene.s_packing, "pixel_margin_resolutions", text="Image resolutions")


class ShotpackerSeparateOverlapping(bpy.types.Operator):
    bl_idname = "uv.spacker_separateoverlapping"
    bl_label = "Separate overlapping UV islands"
    bl_description = "Move overlapping islands to the right by steps"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        if context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        print("uv_overlap: finding overlapping isles...")
        uv_h = uv.UV()
        isles = uv_h.read_isle_set()
        uv_h.group(isles, "on")
        groups = uv_h.groups

        move_one_x = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        changed = []
        for group in groups:
            mtx = isles[group[0]].trs_matrix
            mtx = np.dot(move_one_x, mtx)
            for count, isle_id in enumerate(group):
                isle = isles[isle_id]

                # Separate overlapping islands by 1 UV
                if count > 0:
                    isle.update_verts(mtx)
                    changed.append(isle)

        with amb_util.Mode_set("OBJECT"):
            for c in changed:
                uv_h.write(c)

        print("uv_overlap: done")

        return {"FINISHED"}


class ShotpackerDebugLines(bpy.types.Operator):
    bl_idname = "uv.spacker_debuglines"
    bl_label = "Draw debug lines"
    bl_description = "Draw debug lines to image"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and bpy.app.debug_value != 0

    def execute(self, context):
        if context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        with amb_util.Profile_this():
            uv_h = uv.UV()
            isles = uv_h.read_isle_set()
            uv_h.group(isles, "on")
            groups = uv_h.groups

            lines = []
            colors = []
            for group in groups:
                for count, isle_id in enumerate(group):
                    l = isles[isle_id].lines
                    lines.extend(l)
                    col = [1.0, 0.0, 1.0]
                    if isles[isle_id].broken:
                        col = [1.0, 0.0, 0.0]
                    colors.extend([col for _ in range(len(l))])
                    for chain in isles[isle_id].chains:
                        for i, s in enumerate(chain):
                            assert chain[i][1][0] == chain[(i + 1) % len(chain)][0][0]
                            assert chain[i][1][1] == chain[(i + 1) % len(chain)][0][1]

                    # check segments
                    # create dict for x and y points
                    # each location must have two entries or segments are invalid

            isles, groups = uv_h.generate_isle_set(bpy.context.scene.s_packing.margin / 100.0)
            for group in groups:
                isle_id = group[0]
                l = isles[isle_id].lines
                lines.extend(l)
                colors.extend([[0.0, 1.0, 0.0] for _ in range(len(l))])
                for chain in isles[isle_id].chains:
                    for i, s in enumerate(chain):
                        assert chain[i][1][0] == chain[(i + 1) % len(chain)][0][0]
                        assert chain[i][1][1] == chain[(i + 1) % len(chain)][0][1]

            # with amb_util.Mode_set("OBJECT"):
            #     for c in changed:
            #         uv_h.write(c)

            # for group in groups:
            #     isle = isles[group[0].isle_id]

            # if target image exists, change the size to fit
            xs = bpy.context.scene.s_packing.tex_width
            ys = bpy.context.scene.s_packing.tex_height
            img_name = "shotpacker_debug_lines.png"
            if img_name in bpy.data.images:
                bpy.data.images[img_name].scale(xs, ys)
                image = bpy.data.images[img_name]
            else:
                image = bpy.data.images.new(img_name, width=xs, height=ys)

            pixels = np.zeros((ys, xs, 4))
            pixels[:, :, 3] = 1.0

            import moderngl

            ctx = moderngl.create_context()
            prog = ctx.program(
                vertex_shader="""
                    #version 330

                    in vec2 in_vert;
                    in vec3 in_color;

                    out vec3 v_color;

                    void main() {
                        v_color = in_color;
                        gl_Position = vec4((in_vert * 2.0) - vec2(1.0, 1.0), 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330

                    in vec3 v_color;
                    out vec3 f_color;

                    void main() {
                        f_color = v_color;
                    }
                """,
            )

            x0 = np.zeros(len(lines) * 2)
            y0 = np.zeros(len(lines) * 2)

            for i, l in enumerate(lines):
                x0[i * 2 + 0] = l[0][0]
                x0[i * 2 + 1] = l[1][0]

            for i, l in enumerate(lines):
                y0[i * 2 + 0] = l[0][1]
                y0[i * 2 + 1] = l[1][1]

            r = np.ones(len(colors * 2))
            g = np.ones(len(colors * 2))
            b = np.ones(len(colors * 2))

            for i, c in enumerate(colors):
                r[i * 2], g[i * 2], b[i * 2] = c
                r[i * 2 + 1], g[i * 2 + 1], b[i * 2 + 1] = c

            vertices = np.dstack([x0, y0, r, g, b])

            vbo = ctx.buffer(vertices.astype("f4").tobytes())
            vao = ctx.simple_vertex_array(prog, vbo, "in_vert", "in_color")

            fbo = ctx.simple_framebuffer((xs, ys))
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(moderngl.LINES)

            res = np.frombuffer(fbo.read(), dtype=np.uint8).reshape((*fbo.size, 3)) / 255.0

            pixels[:, :, 0] = res[:, :, 0]
            pixels[:, :, 1] = res[:, :, 1]
            pixels[:, :, 2] = res[:, :, 2]

            image.pixels = pixels.reshape((xs * ys * 4,))

        return {"FINISHED"}


class ShotpackerCollapseSeparated(bpy.types.Operator):
    bl_idname = "uv.spacker_collapseseparated"
    bl_label = "Collapse separated UV islands"
    bl_description = "Move all islands out of the main UV area into it by steps"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None  # and bpy.context.scene.s_packing.overlaptype != "off"

    def execute(self, context):
        if context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        print("uv_collapse: collapsing separated isles...")
        uv_h = uv.UV()
        isles = uv_h.read_isle_set()

        changed = []
        for isle in isles:
            mtx = isle.trs_matrix

            # [val for sublist in list_of_lists for val in sublist]
            # move_x = -int(min(i[0] for l in isle.verts for i in l))
            # move_y = -int(min(i[1] for l in isle.verts for i in l))

            move_x = -int(isle.bounding_box.min_x)
            move_y = -int(isle.bounding_box.min_y)

            if abs(move_x) < 1 and abs(move_y) < 1:
                continue

            mtx = np.dot(np.array([[1, 0, move_x], [0, 1, move_y], [0, 0, 1]]), mtx)

            isle.update_verts(mtx)
            changed.append(isle)

        with amb_util.Mode_set("OBJECT"):
            for c in changed:
                uv_h.write(c)

        print("uv_collapse: done")

        return {"FINISHED"}


class ShotpackerShowBroken(bpy.types.Operator):
    bl_idname = "uv.spacker_showbroken"
    bl_label = "Select broken UV islands"
    bl_description = (
        "Select all islands with invalid geometry.\n"
        "If no invalid geometry found, maintain selection"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        if context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        print("uv_fix: finding broken isles...")

        uv_h = uv.UV()
        isles, _ = uv_h.generate_isle_set(bpy.context.scene.s_packing.margin / 100.0)

        meshes = uv_h.uv_set.managed_meshes
        borken_isles = 0

        bmeshes = []
        for m in meshes:
            bmeshes.append(bmesh.from_edit_mesh(m))
            bmeshes[-1].faces.ensure_lookup_table()

        if not any(i.broken for i in isles):
            # if no broken isles, don't select/unselect anything
            self.report({"INFO"}, "No broken islands found")
            return {"FINISHED"}
        else:
            for isle_i, isle in enumerate(isles):
                # select broken islands, unselect intact ones
                select_it = False
                if isle.broken:
                    borken_isles += 1
                    select_it = True
                    print("broken:", isle_i)
                for poly_id in isle.polys:
                    bmeshes[isle.mesh_id].faces[poly_id].select = select_it

        print("uv_fix: found", borken_isles, "broken")

        # trigger viewport update
        # bpy.context.scene.objects.active = bpy.context.scene.objects.active

        return {"FINISHED"}


class ShotpackerStackSimilar(bpy.types.Operator):
    bl_idname = "uv.spacker_stacksimilar"
    bl_label = "Stack identical UV islands"

    bl_description = (
        "Move identical islands on top of each other.\n"
        "Parameters can be changed after running the operator\n"
        "in the lower left corner"
    )

    bl_options = {"REGISTER", "UNDO"}

    edge_length_margin: bpy.props.FloatProperty(
        name="Edge length margin",
        default=0.005,
        max=1.0,
        min=0.00001,
        description="Edge length margin",
    )

    dot_product_margin: bpy.props.FloatProperty(
        name="Edge angle margin",
        default=0.05,
        max=1.0,
        min=0.00001,
        description="Edge angle margin",
    )

    size_limits: bpy.props.FloatProperty(
        name="Size difference threshold",
        default=1.1,
        max=4.0,
        min=1.0,
        description="Size difference threshold",
    )

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        print("uv_stack: stacking similar isles...")
        uv_h = uv.UV()
        isles = uv_h.read_isle_set()

        moved = [False] * len(isles)

        for i, isle in enumerate(isles):
            if not isle.selected:
                moved[i] = True

        EPS_DOTS = self.dot_product_margin
        EPS_LENS = self.edge_length_margin
        SIZE_LIMIT = self.size_limits

        def _roll_to_match(dots_a, dots_b):
            if len(dots_a) != len(dots_b):
                # print("RTM: Mismatch len")
                return None, None

            # find dots_b offset that matches dots_a
            loc_b = 0
            loop = True
            while loop:
                loop = False
                while loc_b < len(dots_b) and not cgeo.array_kind_of(
                    dots_a[0], dots_b[loc_b], [EPS_DOTS, EPS_LENS]
                ):
                    loc_b += 1

                if loc_b >= len(dots_b):
                    # print("RTM: No starting point found")
                    return None, None

                # rotate dots_b to the matching location
                test_b = dots_b[loc_b:] + dots_b[:loc_b]

                for i in range(len(test_b)):
                    if not cgeo.array_kind_of(dots_a[i], test_b[i], [EPS_DOTS, EPS_LENS]):
                        loop = True
                        loc_b += 1
                        break

            dots_b = test_b

            return dots_b, loc_b

        def _dotstring_test(ica, icb):
            # build and check dot product strings
            dots_a = ica.dot_string()
            dots_b = icb.dot_string()

            len_a = ica.length()
            len_b = icb.length()

            # loop length limit 10%
            if len_a * SIZE_LIMIT < len_b or len_a / SIZE_LIMIT > len_b:
                return None, None

            if not icb.check() or (cgeo.clockwise(icb.chain) != cgeo.clockwise(ica.chain)):
                # print("Invalid B chain")
                return None, None

            dots_b, loc_b = _roll_to_match(dots_a, dots_b)

            if dots_b is None:
                # print("No RTM")
                return None, None

            # TODO: make this ratios, instead of flat values
            # check that angles and lenghts of the edges match
            for i in range(len(dots_a)):
                # dot products, length
                if not cgeo.array_kind_of(dots_a[i], dots_b[i], [EPS_DOTS, EPS_LENS]):
                    # print("Chain mismatch")
                    return None, None

            return dots_b, loc_b

        def _move_to_point(ax, ay, isl_b, ca, cb):
            v0 = np.array([ca[0][1][0] - ca[0][0][0], ca[0][1][1] - ca[0][0][1]])
            v1 = np.array([cb[1][0] - cb[0][0], cb[1][1] - cb[0][1]])

            isl_b.rotate_around(np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1)), ax, ay)
            isl_b.set_location(isl_a.minx, isl_a.miny)

        changed = []

        for idx_a, isl_a in enumerate(isles):
            if moved[idx_a]:
                continue

            for idx_b, isl_b in enumerate(isles[idx_a + 1 :], start=idx_a + 1):
                if moved[idx_b]:
                    continue

                skip = False

                # chain count
                if len(isl_a.chains) != len(isl_b.chains):
                    # print("Break: numchains")
                    continue

                # chain lengths
                for i in range(len(isl_a.chains)):
                    if len(isl_a.chains[i]) != len(isl_b.chains[i]):
                        skip = True
                        break
                if skip is True:
                    # print("Break: lenchains")
                    # print(len(isl_a.chains), len(isl_b.chains))
                    # print("a chains")
                    # for i in isl_a.chains:
                    #     print(len(i), end=", ")
                    # print("\nb chains")
                    # for i in isl_b.chains:
                    #     print(len(i), end=", ")
                    # print()
                    continue

                # print("---")

                # verts
                # if len(isl_a.verts) != len(isl_b.verts):
                #     continue

                # for i in range(len(isl_a.verts)):
                #     if len(isl_a.verts[i]) != len(isl_b.verts[i]):
                #         skip = True
                #         break
                # if skip is True:
                #     continue

                ica = isl_a.chains[0]
                icb = isl_b.chains[0]

                cl_ica = ica.length()
                cl_icb = icb.length()
                if not cgeo.kind_of_ratio(cl_ica, cl_icb, EPS_LENS):
                    # print("Break: total len ratio")
                    continue

                # coincident points
                mirror = False
                dots_b, loc_b = _dotstring_test(ica, icb)
                # print(cgeo.repr_chain(icb))
                if loc_b is None:
                    # try if mirrored
                    mirror = True
                    # print(cgeo.repr_chain(icb))
                    icb = icb.mirror()
                    icb = icb.reverse()
                    # print(cgeo.repr_chain(icb))
                    dots_b, loc_b = _dotstring_test(ica, icb)
                    if loc_b is None:
                        skip = True
                        # print(cl_ica, cl_icb, cgeo.chain_length(icb))
                        # print(cgeo.repr_chain(ica))
                        # print(cgeo.repr_chain(icb))
                        # print("...")
                        # print(cgeo.dot_string(ica))
                        # print(cgeo.dot_string(icb))
                        # print("Break: dot strings")
                        continue
                    # else:
                    #     print("-> ", loc_b)
                    #     print(dots_b)
                    #     dots_a, len_a = cgeo.dot_string(ica)
                    #     print(dots_a)
                # print(cgeo.repr_chain(icb))

                # if island already in position, don't change
                # if cgeo.kind_of(ax, bx, uv_margin) and cgeo.kind_of(ay, by, uv_margin) and \
                #     cgeo.kind_of(isl_a.minx, isl_b.minx, uv_margin) and
                #     cgeo.kind_of(isl_a.miny, isl_b.miny, uv_margin):
                #     skip = True

                # TODO: rotate to MAR, using convex hull edge iteration
                #       only 8 quardrants possible for matching (rotations/flip)
                #       match by (average_of_vert_locations - bounding_box_center)

                # stack identical island
                if not skip:
                    # cx = (isl_a.minx + isl_a.maxx)/2
                    # cy = (isl_a.miny + isl_a.maxy)/2
                    # nx = cx - (isl_b.maxx - isl_b.minx)/2
                    # ny = cy - (isl_b.maxy - isl_b.miny)/2

                    icb.chain = icb.chain[loc_b:] + icb.chain[:loc_b]

                    ax, ay = ica.chain[0][0]
                    bx, by = icb.chain[0][0]

                    # if mirror:
                    #     continue

                    # rotate to match
                    if (
                        dots_b is not None
                    ):  # and not (cgeo.kind_of(ax, bx, 0.0001) and cgeo.kind_of(ay, by, 0.0001)):
                        if mirror:
                            isl_b.flipx()

                        isl_b.scale_from(ica.length() / icb.length())

                        # diag_idx = ica.longest_diagonal()
                        diag_idx = 2

                        ac = ica.chain
                        bc = icb.chain

                        v0 = np.array(
                            [ac[diag_idx][0][0] - ac[0][0][0], ac[diag_idx][0][1] - ac[0][0][1]]
                        )
                        v1 = np.array(
                            [bc[diag_idx][0][0] - bc[0][0][0], bc[diag_idx][0][1] - bc[0][0][1]]
                        )

                        isl_b.rotate_around(cgeo.vec_angle(v0, v1), ax, ay)
                        isl_b.scale_xy(
                            isl_a.bounding_box.width / isl_b.bounding_box.width,
                            isl_a.bounding_box.height / isl_b.bounding_box.height,
                        )
                        isl_b.set_location(isl_a.bounding_box.min_x, isl_a.bounding_box.min_y)

                        isl_b.update_verts(isl_b.trs_matrix)
                        changed.append(isl_b)

                        moved[idx_b] = True

        with amb_util.Mode_set("OBJECT"):
            for c in changed:
                uv_h.write(c)

        print(sum(moved))

        print("uv_stack: done")

        return {"FINISHED"}


class ShotpackerRotateMAR(bpy.types.Operator):
    bl_idname = "uv.spacker_rotatemar"
    bl_label = "Rotate to MAR"
    bl_description = "Rotates selected islands to minimum bounding box"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        print("uv_mar: start")
        uv_h = uv.UV()
        isles = uv_h.read_isle_set()

        for i, isle in enumerate(isles):
            if not isle.selected:
                continue

            isle.rotate_mar()
            isle.update_verts(isle.trs_matrix)
            uv_h.write(isle)

        print("uv_mar: done")

        return {"FINISHED"}


class ShotpackerShowArea(bpy.types.Operator):
    bl_idname = "uv.spacker_showarea"
    bl_label = "Calculate and display used UV area percentage"
    bl_description = "Calculate UV area and the number of islands"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        if context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        bpy.context.scene.s_packing.pack_running = False

        with amb_util.Mode_set("EDIT"):
            uv_set = uv.UVSet()
            area, islecount = uv_set.get_stats()

            context.scene.s_packing.uv_island_count = repr(islecount)
            context.scene.s_packing.uv_area_size = repr(round(area * 100, 2)) + "%"
            print("uv_area: done.")

        return {"FINISHED"}


class ShotpackerGetDimsEditor(bpy.types.Operator):
    bl_idname = "uv.spacker_dims_from_area"
    bl_label = "From editor"
    bl_description = "Looks for texture dimensions in the topmost UV editor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        dims = tools.get_teximage_dims(context)
        if dims[0] != None:
            context.scene.s_packing.tex_width = dims[0]
            context.scene.s_packing.tex_height = dims[1]
        return {"FINISHED"}


class ShotpackerGetDimsObject(bpy.types.Operator):
    bl_idname = "uv.spacker_dims_from_object"
    bl_label = "From object"
    bl_description = "Looks for texture dimensions in the selected object"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        dims = None
        with amb_util.Mode_set("OBJECT"):
            obj = context.object
            for slot_mat in obj.material_slots:
                if slot_mat.material:
                    # Check for traditional texture slots in material
                    for slot_tex in slot_mat.material.texture_paint_slots:
                        if slot_tex and slot_tex.texture and hasattr(slot_tex.texture, "image"):
                            dims = slot_tex.texture.image.size
                    # Check if material uses Nodes
                    if hasattr(slot_mat.material, "node_tree"):
                        if slot_mat.material.node_tree:
                            for node in slot_mat.material.node_tree.nodes:
                                if type(node) is bpy.types.ShaderNodeTexImage:
                                    if node.image:
                                        dims = node.image.size
        if dims != None:
            context.scene.s_packing.tex_width = dims[0]
            context.scene.s_packing.tex_height = dims[1]
        return {"FINISHED"}


class ShotpackerMarkFlipped(bpy.types.Operator):
    bl_idname = "uv.spacker_markflipped"
    bl_label = "Select flipped UV"
    bl_description = "Select flipped UV"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and context.scene.tool_settings.use_uv_select_sync == False
        )

    def execute(self, context):
        if context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        with amb_util.Mode_set("EDIT"):
            uv_h = uv.UV()
            isles = uv_h.read_isle_set()

            for isle in isles:
                mesh = uv_h.uv_set.managed_meshes[isle.mesh_id]
                with amb_bm.Bmesh_from_edit(mesh) as bm:
                    uv_layer = bm.loops.layers.uv.verify()
                    for fi in isle.polys:
                        f = bm.faces[fi]
                        uv_poly = []
                        for l in f.loops:
                            # l[uv_layer].select = False
                            uv_poly.append(tuple(l[uv_layer].uv))
                        cw = cgeo.poly_clockwise(uv_poly)
                        if cw:
                            for l in f.loops:
                                l[uv_layer].select = True

        return {"FINISHED"}


class ShotpackerSelectSelfOverlap(bpy.types.Operator):
    bl_idname = "uv.spacker_selectselfoverlap"
    bl_label = "Select self overlapping UV"
    bl_description = "Select self overlapping UV per island"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def execute(self, context):
        # TODO:
        """
        * Find all edges, connect all edges to faces
        * Sort edges according to X, the edge itself, and all edges among themselves
        * Normal sweepline collision test
        * For all edges that collide, test face collision
        * Select all faces that collide, unselect ones that do not
        problem: not all overlapping UV has self colliding edges

        alternatively:

        * Walk along the island UV polys and don't travel to faces that overlap
        * Overlap:
            A. new edges intersect with old edges
            B. new vertices inside old faces, need to test only the neighbours of the current poly
        * If there is any overlap, rip the produced new island
        * Repeat

        third alternative:

        * Assumptions: all UV polygons are convex
        * Inset all polygons by a very tiny amount
        * Result: any polygons that overlap are self overlap
        * Inner algorithm: test new polygon for intersections on previous polygons
            Possible optimizations:
                100 x 100 grid = 10000 dictionaries
            Is it worth doing bouding box test? Try it later and benchmark
            Separating Axis Theorem (all are convex):
                For any tested edge all vertices of the two polygons are on opposite sides
        """

        print("uv_self_overlap: start")

        with amb_util.Mode_set("EDIT"), amb_util.Profile_this(), amb_bm.Bmesh_from_edit(
            context.active_object.data
        ) as bm:
            uv_h = uv.UV()
            isles = uv_h.read_isle_set()
            uv_layer = bm.loops.layers.uv.verify()
            mask = np.ones(len(bm.faces), dtype=np.bool)

            ASIZE = 200

            def make_edge(e, f):
                loops = [i for i in e.link_loops if i.face == f]
                loops.append(loops[0].link_loop_next)

                assert len(loops) == 2

                uv_edge = (tuple(loops[0][uv_layer].uv), tuple(loops[1][uv_layer].uv))
                if uv_edge[0][0] > uv_edge[1][0]:
                    uv_edge = (uv_edge[1], uv_edge[0])

                return uv_edge

            def uv_overlap(e, f, edges, edge_ids, intersect, accel):
                uv_edge = make_edge(e, f)
                edge_ids.add(e.index)
                edges.add(uv_edge)

                ymin = int(uv_edge[0][1] * ASIZE)
                ymax = int(uv_edge[1][1] * ASIZE)
                if ymax < ymin:
                    ymin, ymax = ymax, ymin

                xmin = int(uv_edge[0][0] * ASIZE)
                xmax = int(uv_edge[1][0] * ASIZE)

                if ymin < 0:
                    ymin = 0
                if xmin < 0:
                    xmin = 0
                if ymax > ASIZE:
                    ymax = ASIZE
                if xmax > ASIZE:
                    xmax = ASIZE

                gather = set()
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        gather |= accel[x][y]
                        accel[x][y].add(uv_edge)

                for x in gather:
                    if (
                        x[1][0] > uv_edge[0][0]
                        and x[0][0] < uv_edge[1][0]
                        and uv_edge[0] != x[0]
                        and uv_edge[1] != x[1]
                        and uv_edge[1] != x[0]
                        and uv_edge[0] != x[1]
                        and bpy_intersect(uv_edge[0], uv_edge[1], x[0], x[1])
                    ):
                        intersect.append(f)
                        return False
                return True

            accel = []
            for i in range(ASIZE + 1):
                accel.append([set() for _ in range(ASIZE + 1)])

            for islei, isle in enumerate(isles):
                # edges = defaultdict(list)
                # for fi in isle.polys:
                #     f = bm.faces[fi]
                #     uv_poly = []
                #     for l in f.loops:
                #         uv1 = tuple(l[uv_layer].uv)
                #         uv_poly.append(uv1)
                #     uvp_len = len(uv_poly)
                #     for ui, u in enumerate(uv_poly):
                #         edge = (u, uv_poly[(ui + 1) % uvp_len])
                #         if edge[0][0] > edge[1][0]:
                #             edge = (edge[1], edge[0])
                #         edges[edge].append(f.loops[ui])

                # np_edges = np.empty((len(edges), 2, 2), dtype=np.float)
                # floops = []
                # for ti, t in enumerate(edges.items()):
                #     k, v = t
                #     np_edges[ti] = np.array(k)
                #     floops.append(v)

                # vec = np_edges[:, 1] - np_edges[:, 0]
                # vecn = np.linalg.norm(vec)
                # vec /= vecn
                # vec *= 0.00001

                # np_edges[:, 0] += vec
                # np_edges[:, 1] -= vec

                # for ei, e in enumerate(np_edges):
                #     for e2i in range(0, ei):
                #         e2 = np_edges[e2i]
                #         if bpy_intersect(e[0], e[1], e2[0], e2[1]):
                #             for l in floops[ei]:
                #                 l[uv_layer].select = True
                #                 l.link_loop_next[uv_layer].select = True

                #             for l in floops[e2i]:
                #                 l[uv_layer].select = True
                #                 l.link_loop_next[uv_layer].select = True

                edges = set()
                edge_ids = set()
                intersect = []

                # for d0 in accel:
                #     for d1 in d0:
                #         d1.clear()

                for f in isle.polys:
                    mask[f] = False

                result, new_edges = amb_bm.traverse_faces(
                    bm,
                    lambda e, f: uv_overlap(e, f, edges, edge_ids, intersect, accel),
                    mark_face=False,
                    mask=mask,
                )

                if len(intersect) > 0:
                    for r in intersect:
                        for l in r.loops:
                            l[uv_layer].select = True

                for f in isle.polys:
                    mask[f] = True

                print("isle ", islei, len(edges), np.sum(new_edges))

            amb_bm.grow_uv_to_faces(bm, uv_layer)

        print("uv_self_overlap: done")

        return {"FINISHED"}


class ShotpackerCloneUVs(bpy.types.Operator):
    bl_idname = "uv.spacker_clone_uvs"
    bl_label = "Clone UVs of objects"
    bl_description = "Clone UVs of objects"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and context.scene.tool_settings.use_uv_select_sync == False
        )

    def execute(self, context):
        if context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        objs = bpy.context.selected_editable_objects

        for o in objs:
            if len(o.data.uv_layers) != 1:
                print("Each object has to have one existing UV map.")
                print(o.name, "has", len(o.data.uv_layers), "layers.")
                return {"CANCELLED"}

        print("UV map test ok.")

        spacker_name = "spacker.temp_layer"

        for obj in objs:
            print(obj.name)

            # remove old
            for l in obj.data.uv_layers:
                if spacker_name in l.name:
                    obj.data.uv_layers.remove(l)

            # add new
            obj.data.uv_layers.new(name=spacker_name)
            obj.data.uv_layers[spacker_name].active = True
            obj.data.uv_layers[spacker_name].active_render = True
            print(obj.data.uv_layers.keys())

        print("Modified", len(objs), "objects.")

        return {"FINISHED"}


class ShotpackerRemoveTempUVs(bpy.types.Operator):
    bl_idname = "uv.spacker_remove_temp_uvs"
    bl_label = "Remove temp UVs"
    bl_description = "Remove temp UVs"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and context.scene.tool_settings.use_uv_select_sync == False
        )

    def execute(self, context):
        objs = bpy.context.selected_editable_objects
        spacker_name = "spacker.temp_layer"
        for obj in objs:
            print(obj.name)

            # remove old
            for l in obj.data.uv_layers:
                if spacker_name in l.name:
                    obj.data.uv_layers.remove(l)

        print("Modified", len(objs), "objects.")

        return {"FINISHED"}


class ShotpackerMaterialsToUDIM(bpy.types.Operator):
    bl_idname = "uv.spacker_materials_to_udim"
    bl_label = "Material UV to UDIM"
    bl_description = "Move each material UV to its own UDIM tile"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and context.scene.tool_settings.use_uv_select_sync == False
        )

    def execute(self, context):
        spacker_name = "spacker.temp_layer"

        objs = bpy.context.selected_editable_objects
        mat_uv = {}
        mat_size = {}
        count = 0
        for obj in objs:
            for slot_mat in obj.material_slots:
                if (
                    slot_mat.material
                    and hasattr(slot_mat.material, "node_tree")
                    and slot_mat.material.node_tree
                ):
                    if slot_mat.material.name not in mat_uv:
                        mat_uv[slot_mat.material.name] = count
                        count += 1

                    # find biggest image size in node tree
                    for node in slot_mat.material.node_tree.nodes:
                        if type(node) is bpy.types.ShaderNodeTexImage and node.image:
                            if slot_mat.material.name not in mat_size:
                                mat_size[slot_mat.material.name] = 0
                            if node.image.size[0] > mat_size[slot_mat.material.name]:
                                mat_size[slot_mat.material.name] = node.image.size[0]

        print("Materials:")
        for k, v in mat_uv.items():
            print("name:", k, "index:", v, "size:", mat_size[k])

        max_size = max(mat_size.values())

        if len(mat_uv.keys()) > 0:
            for obj in objs:
                # bm = bmesh.from_edit_mesh(obj.data)
                with amb_bm.Bmesh_from_edit(obj.data) as bm:
                    # uv_layer = bm.loops.layers.uv.get(obj.data.uv_layers[0].name)
                    uv_layer = bm.loops.layers.uv.get(spacker_name)
                    indices = set()
                    for f in bm.faces:
                        indices.add(f.material_index)
                        mat_name = obj.material_slots[f.material_index].material.name
                        mat_pos = mat_uv[mat_name]
                        scale = mat_size[mat_name] / max_size
                        # scale = 1.0
                        for l in f.loops:
                            cur_x = int(l[uv_layer].uv[0] - 1000.0) + 999
                            cur_y = int(l[uv_layer].uv[1] - 1000.0) + 999
                            new_x = mat_pos % 10
                            new_y = int(mat_pos / 10)
                            cur_uv = l[uv_layer].uv
                            l[uv_layer].uv = mathutils.Vector(
                                [
                                    new_x + (cur_uv[0] - cur_x) * scale,
                                    new_y + (cur_uv[1] - cur_y) * scale,
                                ]
                            )

            print(indices)

        return {"FINISHED"}


class ShotpackerUVTransfer(bpy.types.Operator):
    bl_idname = "uv.spacker_uv_transfer"
    bl_label = "Transfer UV"
    bl_description = "Transfer texture data across UVs"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and context.scene.tool_settings.use_uv_select_sync == False
        )

    def temp(self, context):

        # TODO: create easier image output (pick 2^n image size and prefix)

        amb_util.memorytrace_start()

        # with amb_util.Profile_this():
        ctx = moderngl.create_context()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        prog = ctx.program(
            vertex_shader="""
                #version 330
                //#version 400  -- querylod

                in vec2 in_vert;
                in vec3 in_color;

                out vec3 v_color;

                void main() {
                    v_color = in_color;
                    gl_Position = vec4((in_vert * 2.0) - vec2(1.0, 1.0), 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                //#version 400

                uniform sampler2D Texture;

                in vec3 v_color;
                out vec4 f_color;

                void main() {
                    //float mipmapLevel = textureQueryLod(Texture, v_color.rg).x;
                    f_color = vec4(texture(Texture, v_color.rg, 0).rgb, 1.0);
                }
            """,
        )

        search_text = context.scene.s_packing.uvt_match_string
        spacker_name = "spacker.temp_layer"

        objs = bpy.context.selected_editable_objects
        obj_tris = defaultdict(list)
        obj_locs = defaultdict(list)
        for obj in objs:
            # print(obj.name)
            sel_image = None
            for slot_mat in obj.material_slots:
                if slot_mat.material and hasattr(slot_mat.material, "node_tree"):
                    if slot_mat.material.node_tree:
                        for node in slot_mat.material.node_tree.nodes:
                            if type(node) is bpy.types.ShaderNodeTexImage and node.image:
                                if search_text in node.image.name:
                                    sel_image = node.image
            if sel_image is not None:
                # obj.data.uv_layers[1].active = True
                # obj.data.uv_layers[1].active_render = True
                # print(sel_image.name)
                bm = bmesh.from_edit_mesh(obj.data)
                bmesh.ops.triangulate(
                    bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY"
                )
                uv_layer_target = bm.loops.layers.uv.get(spacker_name)
                uv_layer_source = bm.loops.layers.uv.get(obj.data.uv_layers[0].name)
                for f in bm.faces:
                    assert len(f.loops) == 3
                    obj_tris[sel_image.name].append([tuple(l[uv_layer_target].uv) for l in f.loops])
                    obj_locs[sel_image.name].append([tuple(l[uv_layer_source].uv) for l in f.loops])

        print("Modified", len(objs), "objects.")

        tar_image = context.scene.s_packing.uvt_target_image
        xs, ys = tar_image.size
        pixels = np.zeros((ys, xs, 4))

        fbo = ctx.simple_framebuffer((xs, ys), components=4)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)

        for k in obj_tris.keys():
            print("Very slowly reading images from Blender:", k, len(obj_tris[k]))
            img = bpy.data.images[k]
            texture = ctx.texture(
                img.size, 4, (np.array(img.pixels[:]) * 255).astype("u1").tobytes()
            )
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.build_mipmaps(base=0, max_level=8)
            texture.use(0)

            l_np = np.array(obj_tris[k])
            alen = l_np.shape[0] * l_np.shape[1]
            x0 = l_np[:, :, 0].reshape((alen,))
            y0 = l_np[:, :, 1].reshape((alen,))

            l_np = np.array(obj_locs[k])
            alen = l_np.shape[0] * l_np.shape[1]
            r = l_np[:, :, 0].reshape((alen,))
            g = l_np[:, :, 1].reshape((alen,))
            b = np.ones(len(g))

            vertices = np.dstack([x0, y0, r, g, b])

            vbo = ctx.buffer(vertices.astype("f4").tobytes())
            vao = ctx.simple_vertex_array(prog, vbo, "in_vert", "in_color")
            vao.render(moderngl.TRIANGLES)

        res = np.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape((*fbo.size, 4)) / 255.0

        pixels[:, :, 0] = res[:, :, 0]
        pixels[:, :, 1] = res[:, :, 1]
        pixels[:, :, 2] = res[:, :, 2]
        pixels[:, :, 3] = res[:, :, 3]

        output = pixels.reshape((xs * ys * 4,))
        print("Very slowly saving image data to Blender...")
        tar_image.pixels = output[:]
        print("done.")

        amb_util.memorytrace_print()

        return {"FINISHED"}

    def execute(self, context):
        search_text = context.scene.s_packing.uvt_match_string
        spacker_name = "spacker.temp_layer"

        objs = bpy.context.selected_editable_objects
        obj_tris = defaultdict(list)
        obj_locs = defaultdict(list)
        for obj in objs:
            mat_image = {}
            for slot_mat in obj.material_slots:
                if (
                    slot_mat.material
                    and hasattr(slot_mat.material, "node_tree")
                    and slot_mat.material.node_tree
                ):
                    for node in slot_mat.material.node_tree.nodes:
                        if (
                            type(node) is bpy.types.ShaderNodeTexImage
                            and node.image
                            and search_text in node.image.name
                        ):
                            mat_image[slot_mat.material.name] = node.image.name

            # process all mats and all images
            # add individual UV tris if their material matches
            if len(mat_image) > 0:
                bm = bmesh.from_edit_mesh(obj.data)
                bmesh.ops.triangulate(
                    bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY"
                )
                uv_layer_target = bm.loops.layers.uv.get(spacker_name)
                uv_layer_source = bm.loops.layers.uv.get(obj.data.uv_layers[0].name)

                id_to_name = {k: v.name for k, v in enumerate(obj.material_slots)}
                for f in bm.faces:
                    assert len(f.loops) == 3
                    img_name = mat_image[id_to_name[f.material_index]]
                    obj_tris[img_name].append([tuple(l[uv_layer_target].uv) for l in f.loops])
                    obj_locs[img_name].append([tuple(l[uv_layer_source].uv) for l in f.loops])

        print("Read", len(objs), "objects.")

        tar_image = context.scene.s_packing.uvt_target_image
        width, height = tar_image.size
        offscreen = gpu.types.GPUOffScreen(width, height)

        vertex_shader = """
            in vec2 in_vert;
            in vec3 in_color;

            out vec3 v_color;

            void main() {
                v_color = in_color;
                gl_Position = vec4((in_vert * 2.0) - vec2(1.0, 1.0), 0.0, 1.0);
            }
        """

        fragment_shader = """
            uniform sampler2D Texture;

            // 0 = any, 1 = srgb
            uniform int colorspace;

            in vec3 v_color;
            out vec4 f_color;

            float srgb(float c) {
                if (c<0.0031308) {
                    c*=12.92;
                } else {
                    c=1.055*pow(c, 1.0/2.4)-0.055;
                }

                return clamp(c, 0.0, 1.0);
            }

            void main() {
                f_color = vec4(texture(Texture, v_color.rg, 0).rgb, 1.0);

                if (colorspace == 1) {
                    f_color.r = srgb(f_color.r);
                    f_color.g = srgb(f_color.g);
                    f_color.b = srgb(f_color.b);
                }
            }
        """

        margin_vertex = """
            in vec2 in_vert;
            in vec3 in_color;

            out vec3 v_color;

            void main() {
                v_color = in_color;
                gl_Position = vec4((in_vert * 2.0) - vec2(1.0, 1.0), 0.0, 1.0);
            }
        """

        margin_fragment = """
            uniform sampler2D Texture;

            out vec4 f_color;

            void main() {
                f_color = vec4(texture(Texture, v_color.rg, 0).rgb, 1.0);
            }
        """

        shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
        margin_shader = gpu.types.GPUShader(vertex_shader, margin_shader)
        shader.bind()
        buffer = bgl.Buffer(bgl.GL_BYTE, width * height * 4)
        with offscreen.bind():
            bgl.glClear(bgl.GL_COLOR_BUFFER_BIT)
            with gpu.matrix.push_pop():
                # reset matrices -> use normalized device coordinates [-1, 1]
                gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(mathutils.Matrix.Identity(4))

                for k in obj_tris.keys():
                    print("Image to GL:", k, len(obj_tris[k]))
                    img = bpy.data.images[k]
                    shader.uniform_int(
                        "colorspace", 1 if img.colorspace_settings.name == "sRGB" else 0
                    )
                    img.gl_load()

                    bgl.glActiveTexture(bgl.GL_TEXTURE0)
                    bgl.glBindTexture(bgl.GL_TEXTURE_2D, img.bindcode)

                    batch = batch_for_shader(
                        shader,
                        "TRIS",
                        {
                            "in_vert": [i for s in obj_tris[k] for i in s],
                            "in_color": [(i + (0.0,)) for s in obj_locs[k] for i in s],
                        },
                    )

                    batch.draw(shader)
                    img.gl_free()

                # glCopyTexImage2D(target, level, internalformat, x, y, width, height, border)

            bgl.glReadBuffer(bgl.GL_BACK)
            bgl.glReadPixels(0, 0, width, height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)

        offscreen.free()
        tar_image.pixels = [i / 255.0 for i in buffer.to_list()]
        print("UV transfer done.")

        return {"FINISHED"}


class ShotgunPackingOperator(bpy.types.Operator):
    bl_idname = "uv.shotgunpack"
    bl_label = "Shotpacker"
    bl_description = "Start the packing process"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    running: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.object is not None

    def modal(self, context, event):
        if self.running is False:
            bpy.context.scene.s_packing.uv_packing_progress = ""
            bpy.context.scene.s_packing.pack_running = False

            self.end_run()

            # TODO: Maintain the mode that the script was started with
            self.cancel(context)
            return {"FINISHED"}

        if (event.type in {"ESC"}) or (
            event.type in {"LEFTMOUSE"} and event.value == "RELEASE" or event.value == "PRESS"
        ):
            print("Escape pressed.")
            with amb_util.Mode_set("OBJECT"):
                self.packer.cancel()

            self.running = False

            # critical memory leak fix
            del self.packer

            bpy.context.scene.s_packing.uv_packing_progress = ""
            bpy.context.scene.s_packing.pack_running = False

            self.cancel(context)
            return {"FINISHED"}

        if event.type == "TIMER":
            t0 = time.time()
            pt = True
            # 1 second update interval
            print(
                "shotpacker packing active at {} in {:.2f}s".format(
                    bpy.context.scene.s_packing.uv_packing_progress, time.time() - self.start_time
                )
            )
            while pt and (time.time() - t0) < 1.0:
                pt = self.packer.pack()

            if pt is True:
                for area in bpy.context.screen.areas:
                    area.tag_redraw()
            else:
                self.running = False

        return {"PASS_THROUGH"}

    def init_run(self):
        # print("Prepacking...")
        # bpy.ops.uv.average_islands_scale()
        # bpy.ops.uv.pack_islands(margin=0.001)

        self.start_time = time.time()

        print("Starting packing...")
        self.packer = packer.Packing(uv.UV())

    def end_run(self):
        with amb_util.Mode_set("OBJECT"):
            # write data out
            if self.packer.iterate:
                if self.packer.shuffle_solutions == 0:
                    self.report({"INFO"}, "No better UV solutions found. Current unchanged.")
                self.packer.write_saved()
            else:
                self.packer.write()

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics("lineno")

        # print("tracemalloc top 10")
        # for stat in top_stats[:10]:
        #     print(stat)

        self.end_time = time.time()
        print("End of packing. Time: {:.3f} seconds".format(self.end_time - self.start_time))

        # critical memory leak fix
        del self.packer

    def execute(self, context):
        print("spack execute")
        if bpy.context.object is None:
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        if bpy.context.object.data.uv_layers.active is None:
            self.report({"ERROR"}, "No active UV layers found")
            return {"CANCELLED"}

        if bpy.context.scene.s_packing.pack_running:
            self.report({"ERROR"}, "Shotpacker already running")
            return {"CANCELLED"}

        if self.running is False:
            print("spack running")
            bpy.context.scene.s_packing.pack_running = True
            # start run
            self.running = True
            self.init_run()
            bpy.context.scene.s_packing.uv_packing_progress = "Starting"

            wm = context.window_manager
            self._timer = wm.event_timer_add(time_step=0.01, window=context.window)
            wm.modal_handler_add(self)

            return {"RUNNING_MODAL"}
        else:
            return {"CANCELLED"}

    def cancel(self, context):
        bpy.context.scene.s_packing.uv_packing_progress = ""
        bpy.context.scene.s_packing.pack_running = False
        context.window_manager.event_timer_remove(self._timer)


class UVS_PT_PackerPanelStats(bpy.types.Panel):
    """Shotpacker UV Stats Panel"""

    bl_label = "Shotpacker Stats"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Shotpacker"

    def draw(self, context):
        layout = self.layout

        if context.scene.s_packing.uv_island_count != "":
            row = layout.row()
            row.label(text="Islands: " + context.scene.s_packing.uv_island_count, icon="QUESTION")

            row = layout.row()
            row.label(text="Area: " + context.scene.s_packing.uv_area_size, icon="QUESTION")

        row = layout.row()
        row.operator(ShotpackerShowArea.bl_idname, text="Update stats", icon="BORDERMOVE")


class UVS_PT_PackerPanelTools(bpy.types.Panel):
    """Shotpacker Tools Panel"""

    bl_label = "Shotpacker Tools"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Shotpacker"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout

        if bpy.context.scene.tool_settings.use_uv_select_sync == True:
            row = layout.row()
            row.label(text="UV sync on. Some features disabled.")

        row = layout.row()
        row.label(text="Stacking:")
        row = layout.row()
        row.operator(ShotpackerSeparateOverlapping.bl_idname, icon="FORWARD")
        row = layout.row()
        row.operator(ShotpackerCollapseSeparated.bl_idname, icon="BACK")
        row = layout.row()
        row.operator(ShotpackerStackSimilar.bl_idname, icon="ONIONSKIN_ON")

        row = layout.row()
        row.label(text="Debug:")
        row = layout.row()
        row.operator(ShotpackerShowBroken.bl_idname, icon="X")

        row = layout.row()
        row.operator(ShotpackerMarkFlipped.bl_idname)

        # row = layout.row()
        # row.operator(ShotpackerSelectSelfOverlap.bl_idname)

        if bpy.app.debug_value != 0:
            row = layout.row()
            row.operator(ShotpackerDebugLines.bl_idname)


class UVS_PT_PackerPanelOptions(bpy.types.Panel):
    """Shotpacker Options Panel"""

    bl_label = "Shotpacker Options"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Shotpacker"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(context.scene.s_packing, "init_with_mar", text="Init with MAR")

        row = layout.row()
        row.label(text="Performance:")

        row = layout.row()
        row.prop(context.scene.s_packing, "iterations", text="Island iterations")

        # row = layout.row()
        # row.label(text="Island margin accuracy:")
        # row = layout.row()
        # row.prop(context.scene, "s_packing.marginquality", expand=True,
        # text="Island margin quality/performance")
        row = layout.row()
        row.prop(context.scene.s_packing, "pruning", text="Pruning ratio")


class UVS_PT_PackerPanelTransfer(bpy.types.Panel):
    """Shotpacker UV Transfer Panel"""

    bl_label = "Shotpacker UV Transfer"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Shotpacker"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.operator(ShotpackerCloneUVs.bl_idname)
        row = layout.row()
        row.operator(ShotpackerMaterialsToUDIM.bl_idname)

        box = layout.box()
        col = box.column(align=True)

        row = col.row()
        row.prop(context.scene.s_packing, "uvt_match_string", text="Match string")

        row = col.row()
        row.prop(context.scene.s_packing, "uvt_target_image", text="Target")

        row = box.row()
        row.operator(ShotpackerUVTransfer.bl_idname)
        row = layout.row()
        row.operator(ShotpackerRemoveTempUVs.bl_idname)


class UVS_PT_PackerPanel(bpy.types.Panel):
    """Shotpacker Packing Panel"""

    bl_label = "Shotpacker Packing"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Shotpacker"

    def draw(self, context):
        layout = self.layout

        row = layout.row()

        if MODULES_LOADED:
            row = layout.row()
            row.scale_y = 2.0
            bt_text = "Pack"

            if bpy.context.scene.s_packing.uv_packing_progress != "":
                bt_text = "{} (Esc or mouse button to stop)".format(
                    bpy.context.scene.s_packing.uv_packing_progress
                )

            row.operator(ShotgunPackingOperator.bl_idname, text=bt_text)

            row = layout.row()
            row.prop(context.scene.s_packing, "iterate", text="Keep iterating")

            col = layout.column()
            row = col.row(align=True)
            # row.label(text="Packing quality:")
            row.prop(context.scene.s_packing, "quality", text="")
            row.prop(context.scene.s_packing, "rotations", text="")
            row.prop(context.scene.s_packing, "flip", text="Mirror")

            col = layout.column()
            row = col.row(align=True)
            row.label(text="Current resolution:")
            row = col.row(align=True)
            row.prop(context.scene.s_packing, "tex_width", text="Width")
            row.prop(context.scene.s_packing, "tex_height", text="Height")
            # if not bpy.context.scene.s_packing.use_pixel_margins:
            #     row.prop(context.scene.s_packing, "margin", text="Margin (%)")
            # else:
            row.prop(context.scene.s_packing, "pixel_margin", text="Pixels")
            row = col.row(align=True)
            row.label(text="Change resolution:")
            row = col.row(align=True)
            row.prop(context.scene.s_packing, "pixel_resolution_enums", text="")
            row.operator(ShotpackerGetDimsEditor.bl_idname, text="Editor")
            row.operator(ShotpackerGetDimsObject.bl_idname, text="Object")

            # row = layout.row(align=True)

            row = layout.row()
            row.label(text="Island selection behavior:")
            row = layout.row()
            row.prop(context.scene.s_packing, "selectbehavior", expand=True)

            row = layout.row()
            row.label(text="Overlap detection:")
            row = layout.row()
            row.prop(context.scene.s_packing, "overlaptype", expand=True)

            # row = layout.row()
            # row.prop(context.scene, "s_packing.repair", text="Try repair broken islands")

        else:
            row.label(text="Required library not found.")
            row = layout.row()
            # row.label(text="Please install all required modules")
            # row = layout.row()
            # row.label(text="from the Blender preferences ->")
            # row = layout.row()
            # row.label(text="Add-ons -> Shotpacker preferences.")
            row.operator(ShotpackerInstallLibraries.bl_idname, text="Install required libraries")


class ShotpackerPropertyGroup(bpy.types.PropertyGroup):
    uv_packing_progress: bpy.props.StringProperty(
        name="uv_packing_progress", description="UV Packing Progress"
    )

    iterations: bpy.props.IntProperty(
        name="s_packing.iterations",
        default=400,
        min=100,
        max=2000,
        description="Different vertical and horizontal positions tested per island",
    )

    rotations: bpy.props.EnumProperty(
        items=[
            ("1", "1 (360°)", ""),
            ("2", "2 (180°)", ""),
            ("4", "4 (90°)", ""),
            ("8", "8 (45°)", ""),
            ("16", "16 (22.5°)", ""),
            ("32", "32 (11.3°)", ""),
            ("64", "64 (5.6°)", ""),
            ("128", "128 (2.8°)", ""),
            ("256", "256 (1.4°)", ""),
        ],
        name="Rotational positions",
        default="4",
        description="Number of different island rotations tested per location",
    )

    # --- prefs
    # use_pixel_margins: bpy.props.BoolProperty(
    #     name="s_packing.use_pixel_margins",
    #     default=True,
    #     description="Use pixel margin instead of ratios",
    # )

    pixel_margin_resolutions: bpy.props.StringProperty(
        name="s_packing.pixel_margin_resolutions",
        default="64,128,256,512,1024,2048,4096,8192",
        description="Texture resolution choices for pixel margin setting",
    )

    # -- margins
    margin: bpy.props.FloatProperty(
        name="s_packing.margin",
        default=0.1,
        max=50.0,
        min=0.001,
        description="Percentage of the UV area that is used as margin per island",
    )

    def quality_update(self, context):
        if self.quality == "DRAFT":
            self.pruning = 0.2
            self.iterations = 100
        if self.quality == "GOOD":
            self.pruning = 0.02
            self.iterations = 400
        if self.quality == "HIGH":
            self.pruning = 0.02
            self.iterations = 1000
        if self.quality == "ULTRA":
            self.pruning = 0.0
            self.iterations = 2000

    quality: bpy.props.EnumProperty(
        items=[
            ("DRAFT", "Draft", ""),
            ("GOOD", "Good", ""),
            ("HIGH", "High", ""),
            ("ULTRA", "Ultra", ""),
        ],
        update=quality_update,
        name="Quality",
        default="GOOD",
        description="Packing quality presets.\nChanges island iterations and pruning ratio."
        "\n(In Shotpacker Options).",
    )

    tex_width: bpy.props.IntProperty(
        name="s_packing.tex_width", default=1024, min=1, description="Texture width"
    )

    tex_height: bpy.props.IntProperty(
        name="s_packing.tex_height", default=1024, min=1, description="Texture height"
    )

    pixel_resolution: bpy.props.IntProperty(
        name="s_packing.pixel_margin_resolution", default=1024, min=8, max=65536
    )

    def pixel_margin_update(self, context):
        self.pixel_resolution = min(self.tex_width, self.tex_height)
        self.margin = self.pixel_margin * 100.0 / int(self.pixel_resolution)

    def pixel_resolution_update(self, context):
        self.tex_width = int(self.pixel_resolution_enums)
        self.tex_height = int(self.pixel_resolution_enums)
        self.pixel_resolution = min(self.tex_width, self.tex_height)
        self.margin = self.pixel_margin * 100.0 / int(self.pixel_resolution)

    def pixel_res_dynamic(self, context):
        return [(i, i, "") for i in bpy.context.scene.s_packing.pixel_margin_resolutions.split(",")]

    pixel_resolution_enums: bpy.props.EnumProperty(
        items=pixel_res_dynamic,
        update=pixel_resolution_update,
        name="Image resolutions",
        description="Image resolution the pixel margin setting is based on",
    )

    pixel_margin: bpy.props.IntProperty(
        name="s_packing.pixel_margin",
        default=1,
        max=100,
        min=0,
        description="Number of pixels used as margin per island",
        update=pixel_margin_update,
    )
    # -- margins end

    flip: bpy.props.BoolProperty(
        name="s_packing.flip",
        default=True,
        description="Add mirrored islands to the tested positions",
    )

    pruning: bpy.props.FloatProperty(
        name="s_packing.pruning",
        default=0.00,
        max=1.0,
        min=0.0,
        description="0 = zero simplification for islands (same as input)\n"
        "1 = max simplification for islands (worse packing ratio, faster)\n"
        "0.02 is a nice default if you want more performance",
    )

    selectbehavior: bpy.props.EnumProperty(
        items=[
            ("all", "All", "Pack all UV islands, including hidden and unselected"),
            (
                "selected",
                "Selected",
                "Pack only selected and visible UV islands, maintain scale"
                " and collide on unselected",
            ),
            ("ignore", "Ignore", "Completely ignore unselected and hidden UV islands"),
            ("lock", "Lock", "Lock scale, do not scale anything"),
        ],
        name="s_packing.selectbehavior",
        default="all",
    )

    overlaptype: bpy.props.EnumProperty(
        items=[
            ("off", "Off", "No overlap detection", "CHECKBOX_DEHLT", 1),
            ("on", "On", "Group overlapping islands", "MOD_OPACITY", 2),
        ],
        name="s_packing.overlaptype",
        default="off",
    )

    init_with_mar: bpy.props.BoolProperty(
        name="s_packing.init_with_mar",
        default=False,
        description="Before starting the packing process, rotate islands"
        " to Minimum-Area bounding Rectangle",
    )

    iterate: bpy.props.BoolProperty(
        name="s_packing.iterate",
        default=False,
        description="Continue finding new solutions.\nLeft number is new"
        " solutions found.\nRight number is solutions tried.\n"
        "Can be stopped any time by pressing escape.\n"
        "Stopping will save the best solution found.\n"
        "Will automatically stop when no new solutions found after a while.\n"
        "If no better solution found, it will maintain the original UV",
    )

    pack_running: bpy.props.BoolProperty(name="s_packing.pack_running", default=False)

    uv_island_count: bpy.props.StringProperty(name="s_packing.uv_island_count")
    uv_area_size: bpy.props.StringProperty(name="s_packing.uv_area_size")

    uvt_match_string: bpy.props.StringProperty(name="s_packing.uvt_match_string")
    uvt_target_image: bpy.props.PointerProperty(
        name="s_packing.uvt_target_image", type=bpy.types.Image
    )


# The experimental features are commented out but you can try them if you want
# by uncommenting the classes, and their invocations in the panels

classes = (
    ShotpackerPropertyGroup,
    ShotpackerInstallLibraries,
    # ShotpackerUninstallAccelerator,
    ShotpackerAddonPreferences,
    ShotgunPackingOperator,
    ShotpackerShowBroken,
    ShotpackerShowArea,
    ShotpackerSeparateOverlapping,
    ShotpackerCollapseSeparated,
    ShotpackerStackSimilar,
    ShotpackerMarkFlipped,
    # ShotpackerSelectSelfOverlap,
    ShotpackerRotateMAR,
    ShotpackerGetDimsEditor,
    ShotpackerGetDimsObject,
    ShotpackerDebugLines,
    ShotpackerCloneUVs,
    ShotpackerRemoveTempUVs,
    # ShotpackerUVTransfer,
    ShotpackerMaterialsToUDIM,
    UVS_PT_PackerPanel,
    UVS_PT_PackerPanelStats,
    UVS_PT_PackerPanelTools,
    # UVS_PT_PackerPanelTransfer,
    UVS_PT_PackerPanelOptions,
)


def register():
    print("Shotpacker: Register addon.")

    # if "accelerator" in globals():
    #     accelerator.init_dll()
    #     print("Re-init accelerator DLL.")
    # else:
    #     print("No existing accelerator DLL.")

    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.s_packing = bpy.props.PointerProperty(type=ShotpackerPropertyGroup)


def unregister():
    print("Shotpacker: Unregister addon.")

    # if "accelerator" in globals():
    #     print("Disabling Shotpacker accelerator component...")
    #     accelerator.ffi.dlclose(accelerator.ffilib)
    # else:
    #     print("No Shotpacker accelerator loaded.")

    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.s_packing
