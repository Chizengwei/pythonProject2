"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Created Date: Tuesday, February 5th 2019, 10:18:04 am
Author: Tommi Hyppänen
"""


# import math
import random
import time

from collections import defaultdict
import bpy

# import bmesh
from . import cgeo, island
from .bpy_amb import bbmesh as amb_bm
from .bpy_amb import utils as amb_util

rnd = random.random


class UVSet:
    """ Builds a set of connected (through UV) polygon ids for objects \
        return type: list[object][isle][polygon index] """

    __slots__ = [
        "uv_layers",
        "managed_objects",
        "managed_meshes",
        "isles",
        "edge_continuity",
        "border_segs",
    ]

    def __init__(self):
        # start parsing UV islands from the UV data
        self.uv_layers = []

        _start_0 = time.time()
        self.managed_objects = bpy.context.selected_editable_objects
        # print("Selected editable objects:", len(self.managed_objects))

        self.managed_objects = [i for i in self.managed_objects if type(i.data) == bpy.types.Mesh]
        # print("Found objects:", len(self.managed_objects))

        self.managed_meshes = list(set([o.data for o in self.managed_objects]))
        # print("Found meshes:", len(self.managed_meshes))

        self.isles = [[] for _ in range(len(self.managed_meshes))]
        self.edge_continuity = [[] for _ in range(len(self.managed_meshes))]
        self.border_segs = [[] for _ in range(len(self.managed_meshes))]

        # bpy.ops.uv.select_linked()
        # bpy.ops.uv.select_more()

        for mesh_id, mesh in enumerate(self.managed_meshes):
            # print("mesh:", mesh_id)
            with amb_bm.Bmesh_from_edit(mesh) as bm:
                uv_layer = bm.loops.layers.uv.verify()
                self.uv_layers.append(uv_layer)

                # find continuous edges over UV and mesh
                for i, e in enumerate(bm.edges):
                    if e.is_contiguous and e.is_manifold:
                        # result = amb_bm.edge_same_uv(e, uv_layer)
                        l0, l1 = e.link_loops

                        # ASSUMPTION: loop always goes the same direction (link_loop_next)
                        a0, a1 = l0[uv_layer].uv, l0.link_loop_next[uv_layer].uv
                        b0, b1 = l1[uv_layer].uv, l1.link_loop_next[uv_layer].uv

                        if (a0 == b0 and a1 == b1) or (a0 == b1 and a1 == b0):
                            result = True
                            # ASSUMPTION: <loop0 previous> is another face from <loop1 next>
                            center0 = l0.link_loop_prev[uv_layer].uv
                            center1 = l1.link_loop_next[uv_layer].uv
                            uv0 = l0[uv_layer].uv
                            uv1 = l1[uv_layer].uv
                            uv_vec = uv1 - uv0

                            # Test for UV folds
                            cr1 = uv_vec.cross(center0 - uv0)
                            cr2 = uv_vec.cross(center1 - uv1)
                            if cr1 * cr2 > 0:
                                result = False
                        else:
                            result = False
                    else:
                        result = False

                    # if result == True:
                    #     # check for same side faces (self overlap uv)
                    #     # centers = []
                    #     # for f in e.link_faces:
                    #     #     uvs = []
                    #     #     for l in f.loops:
                    #     #         uvs.append(l[uv_layer].uv)
                    #     #     vsum = uvs[0].copy()
                    #     #     for i in range(len(uvs) - 1):
                    #     #         vsum += uvs[i + 1]
                    #     #     centers.append(vsum / len(uvs))
                    #     # assert len(centers) == 2

                    #     check that faces are on different sides of the uv edge
                    #     uv0, uv1 = [l[uv_layer].uv for l in e.link_loops]
                    #     l0 = e.link_loops[0]
                    #     l1 = e.link_loops[1]
                    #     center0 = l0.link_loop_prev[uv_layer].uv
                    #     center1 = l1.link_loop_next[uv_layer].uv
                    #     uv0 = l0[uv_layer].uv
                    #     uv1 = l1[uv_layer].uv
                    #     uv_vec = uv1 - uv0
                    #     cr1 = uv_vec.cross(center0 - uv0)
                    #     cr2 = uv_vec.cross(center1 - uv1)
                    #     if cr1 * cr2 > 0:
                    #         result = False

                    self.edge_continuity[mesh_id].append(result)

                edge_suv = self.edge_continuity[mesh_id]

                # link edges to uv locations
                # uv_count = defaultdict(int)
                # uv_loc = defaultdict(list)
                for i in range(len(edge_suv)):
                    if not edge_suv[i]:
                        e = bm.edges[i]
                        for f in e.link_faces:
                            uv = []
                            for l in f.loops:
                                if l.vert in e.verts:
                                    uvl = tuple(l[uv_layer].uv)
                                    uv.append(uvl)
                                    # uv_count[uvl] += 1
                                    # uv_loc[uvl].append(l)

                        assert len(uv) == 2
                        self.border_segs[mesh_id].append(((tuple(uv[0]), tuple(uv[1])), e.index))

                # for b in self.border_segs[mesh_id]:
                #     e = bm.edges[b[1]]
                #     for f in e.link_faces:
                #         for l in f.loops:
                #             if l.vert in e.verts:
                #                 l[uv_layer].select = True

                # if exist, move X pinpoint uv to remove singularity overlap
                # for k, v in uv_count.items():
                #     if v > 2:
                #         uvt = uv_loc[k][0][uv_layer].uv
                #         uv_loc[k][0][uv_layer].uv = (uvt[0] + 0.00001, uvt[1])
                #         uv_loc[k][0][uv_layer].select = True

                # traverse UV shells to create islands (collection of faces)
                remaining = set(bm.faces[:])
                while len(remaining) > 0:
                    neighbours = set([next(iter(remaining))])
                    new_faces = set()
                    while len(neighbours) > 0:
                        new_faces |= neighbours
                        step = set()
                        for f in neighbours:
                            if f in remaining:
                                remaining.discard(f)
                                for e in f.edges:
                                    if edge_suv[e.index]:
                                        lf = e.link_faces
                                        step.add(lf[0] if lf[0] != f else lf[1])
                        neighbours = step

                    if len(new_faces) > 0:
                        self.isles[mesh_id].append([i.index for i in new_faces])

        print("\nUV read in {:.3f} seconds".format(time.time() - _start_0))

    def get_loops_from_continuity(self, mesh_i):
        bsegs = self.border_segs[mesh_i]
        loc_to_edge = defaultdict(list)
        remaining = set()
        seg_count = defaultdict(int)
        for ss in bsegs:
            s = ss[0]
            loc_to_edge[s[0]].append(s)
            loc_to_edge[s[1]].append(s)
            seg_count[s[0]] += 1
            seg_count[s[1]] += 1
            remaining.add(s)

        # for k, v in seg_count.items():
        #     if v != 2:
        #         print(k, v)

        # build chains
        result = []
        while len(remaining) > 0:
            chain = []
            seg = remaining.pop()
            loc = seg[1]
            while True:
                chain.append(loc)
                uv_e = [s for s in loc_to_edge[loc] if s != seg]
                if len(uv_e) == 1 and uv_e[0] in remaining:
                    seg = uv_e[0]
                    loc = seg[0] if seg[1] == loc else seg[1]
                    remaining.discard(seg)
                else:
                    break
            result.append(chain)

        # print(len(result))

    def get_stats(self):
        area = 0.0
        islecount = 0

        for mesh_i, mesh in enumerate(self.managed_meshes):
            # print(mesh_i, len(self.isles[mesh_i]))
            islefaces = []
            for i in self.isles[mesh_i]:
                islefaces.append(list(i))

            with amb_bm.Bmesh_from_edit(self.managed_meshes[mesh_i]) as bm:
                uv_layer = bm.loops.layers.uv.verify()
                for isle in islefaces:
                    for f in isle:
                        area += cgeo.poly_area([l[uv_layer].uv for l in bm.faces[f].loops])

            islecount += len(islefaces)
        return area, islecount


class UV:
    __slots__ = ["islands", "texture_ratio", "uv_set", "islefaces", "groups", "grouping_type"]

    def __init__(self):
        self.islands = []

        tex_width = bpy.context.scene.s_packing.tex_width
        tex_height = bpy.context.scene.s_packing.tex_height

        print("Active texture dimensions:", tex_width, tex_height)
        self.texture_ratio = tex_width / tex_height

        # all polys
        print("Parse UV...")
        self.uv_set = UVSet()
        self.islefaces = [list(i) for i in self.uv_set.isles]

        self.grouping_type = bpy.context.scene.s_packing.overlaptype

    def scale_verts_to(self, x, y):
        if self.texture_ratio < 1:
            return x, y / self.texture_ratio
        else:
            return x * self.texture_ratio, y

    def scale_verts_from(self, x, y):
        if self.texture_ratio < 1:
            return x, y * self.texture_ratio
        else:
            return x / self.texture_ratio, y

    def write(self, island):
        for fi, f in enumerate(island.polys):
            mesh = self.uv_set.managed_meshes[island.mesh_id]
            for i, loop_idx in enumerate(mesh.polygons[f].loop_indices):
                x, y = island.verts[fi][i]
                mesh.uv_layers.active.data[loop_idx].uv = self.scale_verts_from(x, y)

    def write_bb(self, island, bm):
        uv_layer = self.uv_set.uv_layers[island.mesh_id]
        for fi, f in enumerate(island.polys):
            for i, l in enumerate(bm.faces[f].loops):
                x, y = island.verts[fi][i]
                l[uv_layer].uv = self.scale_verts_from(x, y)

    def _order_isle_set(self, isles):
        # largest and most complex (bounding box area + negative uv area)
        max_bb = 0.0
        for isle in isles:
            if isle.bounding_box.area > max_bb:
                max_bb = isle.bounding_box.area

        # hw = bpy.context.scene.s_packing.heuristic
        # r = bpy.context.scene.s_packing.randomness
        hw = 0.0
        r = 0.0

        # Island placement order heuristic (bounding box vs. uv area)
        groups = sorted(
            self.groups,
            key=lambda x: 2
            - (1.0 - hw) * isles[x[0]].bounding_box.area
            - hw * isles[x[0]].uv_area
            + rnd() * max_bb * r,
        )

        return isles, groups

    def group(self, isles, grouping_type):
        # group directly overlapping UV islands (matching set of UV coordinates)
        self.groups = []
        print("Grouping:")
        print("  islands...")
        if grouping_type != "off":
            placed = set()
            for i, a in enumerate(isles):
                if i not in placed:
                    self.groups.append([i])
                    placed.add(i)
                    for ji, b in enumerate(isles[i + 1 :], start=i + 1):
                        if ji not in placed and isles[i].check_collision(isles[ji]):
                            # merge an island into an existing group
                            self.groups[-1].append(ji)
                            placed.add(ji)
                            # i = isle A, ji = isle B
                            isles[i].chains = cgeo.clipper_island_union([isles[i], isles[ji]])
                            isles[i].build_lines_from_chains()
            # merge colliding groups
            print("  groups...")
            while True:
                new_groups = []
                zero_new = True
                placed = set()
                for i in range(len(self.groups)):
                    if i not in placed:
                        a = self.groups[i][0]
                        new_groups.append(self.groups[i])
                        placed.add(i)

                        if len(self.groups[i]) > 1:
                            for j in range(i + 1, len(self.groups)):
                                b = self.groups[j][0]
                                if j not in placed and isles[a].check_collision(isles[b]):
                                    zero_new = False

                                    # merge a group into an existing group
                                    new_groups[-1].extend(self.groups[j])
                                    placed.add(j)

                                    isles[a].chains = cgeo.clipper_island_union(
                                        [isles[a], isles[b]]
                                    )
                                    isles[a].build_lines_from_chains()
                if zero_new:
                    break
                print("  .. joined groups")
                self.groups = new_groups

        else:
            for i, a in enumerate(isles):
                self.groups.append([i])
        print("  done.")

    def generate_isle_verts(self, polys, bm, uv_layer):
        selected = False
        visible = False
        verts = []
        uv_ss = bpy.context.scene.tool_settings.use_uv_select_sync
        for f in polys:
            spoly = bm.faces[f]
            tar = []
            for loop in spoly.loops:
                # Scaling needed because Blender considers
                # all width to height ratios as [0..1, 0..1]
                x, y = self.scale_verts_to(float(loop[uv_layer].uv[0]), float(loop[uv_layer].uv[1]))
                tar.append([x, y])

                # Flag whole island selected if any uv vert marked as selected
                lfsel = loop.face.select
                lfhid = loop.face.hide
                lusel = loop[uv_layer].select

                if uv_ss:
                    # if UV sync is on, selected UV = selected mesh faces
                    selected |= lfsel
                else:
                    # UV sync off, selected UV = selected UV on visible mesh selection
                    selected |= lusel & lfsel

                visible |= not lfhid

            verts.append(tar)

        return verts, selected, not visible

    def read_isle_set(self):
        isles = []
        for mesh_i, mesh_f in enumerate(self.islefaces):
            with amb_bm.Bmesh_from_edit(self.uv_set.managed_meshes[mesh_i]) as bm:
                uv_layer = bm.loops.layers.uv.verify()
                for polys in mesh_f:
                    verts, selected, hidden = self.generate_isle_verts(polys, bm, uv_layer)

                    isle = island.Island(polys, verts, selected, hidden)
                    isle.mesh_id = mesh_i
                    # self.uv_set.get_loops_from_continuity(mesh_i)
                    valid = isle.make_chains(cgeo.find_mesh_chains(isle.verts))

                    if valid:
                        isle.build_lines_from_chains()
                        isles.append(isle)
                    else:
                        print("Invalid UV data found. Algorithm result undetermined.")
                        isle._fallback()
                        isles.append(isle)

        return isles

    def generate_isle_set(self, margin):
        isles = self.read_isle_set()

        self.group(isles, self.grouping_type)

        # if any of the merged islands is selected,
        # mark group selected
        # (assume: the first island in group is the group)
        for g in self.groups:
            for i in g:
                if isles[i].selected is True:
                    for j in g:
                        isles[j].selected = True
                    break

        # if any of the merged islands is visible, mark group visible
        for g in self.groups:
            for i in g:
                if isles[i].hidden is False:
                    for j in g:
                        isles[j].hidden = False
                    break

        # run formatting functions
        for i, _ in enumerate(isles):
            isles[i].format_for_packing(margin)
            isles[i].build_lines_from_chains()

        tpp = sum(i.prune_total for i in isles)
        if tpp > 0:
            print("Prune ratio: {0:.2f}%".format(sum(i.prune_result for i in isles) * 100.0 / tpp))

        return self._order_isle_set(isles)
