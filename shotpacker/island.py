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


# import copy
import heapq

# from dataclasses import dataclass
# from collections import defaultdict

import numpy as np

import bpy
from mathutils import geometry as gm

from . import cgeo

# from . import pyclipper

bpy_intersect = gm.intersect_line_line_2d


class BoundingBox:
    """ Y is height """

    __slots__ = [
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "width",
        "height",
        "area",
        "center_x",
        "center_y",
    ]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: float
    height: float
    area: float
    center_x: float
    center_y: float

    def refresh(self, x_values, y_values):
        self.max_x = np.max(x_values)
        self.min_x = np.min(x_values)
        self.max_y = np.max(y_values)
        self.min_y = np.min(y_values)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

        self.area = self.width * self.height

        self.center_x = (self.max_x + self.min_x) / 2
        self.center_y = (self.max_y + self.min_y) / 2

    def AABB(self, other):
        return (
            self.min_x > other.max_x
            or self.min_y > other.max_y
            or self.max_x < other.min_x
            or self.max_y < other.min_y
        )

    def move(self, x, y):
        self.max_x += x
        self.min_x += x
        self.max_y += y
        self.min_y += y
        self.center_x += x
        self.center_y += y


class Island:
    __slots__ = [
        "margin",
        "trs_matrix",
        "verts",
        "polys",
        "mesh_id",
        "picked",
        "selected",
        "hidden",
        "broken",
        "chains",
        "lines",
        "uv_area",
        "prune_ratio",
        "prune_total",
        "prune_result",
        "bounding_box",
    ]

    def __init__(self, polys, verts, selected, hidden):
        self.margin = 0.00001

        self.trs_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # verts matching poly indices
        self.verts = verts
        self.polys = polys

        # self.object_id = 0
        self.mesh_id = 0
        self.picked = False
        self.selected = selected
        self.hidden = hidden

        self.broken = False

        self.prune_ratio = bpy.context.scene.s_packing.pruning

        self.prune_total = 0
        self.prune_result = 0

        self.chains = None
        self.uv_area = 0

        self.bounding_box = BoundingBox()

    def make_chains(self, chains):
        # self.chains = cgeo.find_mesh_chains(self.verts)
        self.chains = chains
        if not self.chains:
            print("No chains found.")
            return False

        self.chains = [c for c in self.chains if len(c) > 2]

        if len(self.chains) == 0:
            return False

        # sort by biggest horizontal bound
        bounds = []
        for c in self.chains:
            minx, _, maxx, _ = c.bounding_box()
            bounds.append(maxx - minx)

        # then set the widest chain to first location
        loc = bounds.index(min(bounds))
        bounds[loc], bounds[0] = bounds[0], bounds[loc]
        self.chains = [i for _, i in sorted(zip(bounds, self.chains), key=lambda x: -x[0])]

        # calculate resulting area
        area = 0.0
        if self.chains and len(self.chains) > 0:
            for i in range(0, len(self.chains)):
                area += cgeo.poly_area([i[0] for i in self.chains[i]])
        self.uv_area = area

        return True

    def format_for_packing(self, margin):
        """ Dilate and prune chains """
        self.margin = margin

        smp = self.margin + self.margin * self.prune_ratio
        min_bb = smp * 4
        if min_bb < 0.01:
            min_bb = 0.01

        # DILATE
        any_bad = False
        for c in self.chains:
            if not c.check():
                any_bad = True
            if any_bad:
                print("invalid chain!")

        if (
            any_bad
            or len(self.chains[0]) < 5
            or self.bounding_box.width < min_bb
            or self.bounding_box.height < min_bb
        ):
            for ch in self.chains:
                ch.dilate(smp)
        else:
            # Bugfix: sometimes clipper_offset() may produce len(self.chains) == 0
            temp_chains = cgeo.clipper_offset(self.chains, smp)
            if len(temp_chains) < 1:
                print("PyClipper: Invalid response")
                for ch in self.chains:
                    ch.dilate(smp)
            else:
                self.chains = temp_chains

        # PRUNE
        for i, c in enumerate(self.chains):
            if len(self.chains[i]) > 10:
                self.prune_total += len(self.chains[i])
                if self.prune_ratio > 0.0 and self.margin > 0.0:
                    self.chains[i].douglas_peucker(self.margin * self.prune_ratio)
                self.prune_result += len(self.chains[i])

    def build_lines_from_chains(self):
        # sort and remove dups
        self.lines = []
        if len(self.chains) == 0:
            print("Shotpacker couldn't build island, trying fallback.")
            self._fallback()
            return

        for ch in self.chains:
            self.lines.extend(ch.chain)

        tlines = [(i[0], i[1]) if i[0] < i[1] else (i[1], i[0]) for i in self.lines]
        tlines = list(set(tlines))
        tlines.sort()

        self.lines = np.empty(shape=(len(tlines), 2, 2), dtype=np.float)

        for i, l in enumerate(tlines):
            self.lines[i] = ((l[0][0], l[0][1]), (l[1][0], l[1][1]))

        self._bb_calc()

        # too small island = fallback
        if (
            self.bounding_box.max_x - self.bounding_box.min_x < 1 / 10000
            or self.bounding_box.max_y - self.bounding_box.min_y < 1 / 10000
        ):
            self._fallback()

    def check_collision(self, other):
        # check bounding boxes
        if self.bounding_box.AABB(other.bounding_box):
            return False

        # check for point inside polygon collision, leftmost point
        def _onepoint(lines_a, lines_b):
            x, y = lines_a[0][0]
            if x > lines_b[0][0][0]:
                # flip flag every time collide with edge
                inside = False

                # determination if inside at center
                true_inside = False

                # +0.000000195845 is here to stop point overlap (make it very very rare)
                iline = ((-100.0, y + 0.000000195845), lines_a[0][0])
                for ol in lines_b:
                    if ol[0][0] > x:
                        break
                        # true_inside = inside
                        # iline = ((100.0, y + 0.000000195845), lines_a[0][0])
                    if bpy_intersect(*iline, *ol):
                        inside = not inside

                # inside must be False after looping through all the edges
                # for the determination to be valid
                # if true_inside and not inside:
                # TODO: still some issues with overlapping (check suz pbr blendswap)
                if inside:
                    return True
            return False

        # # check if self is inside other or other is inside self
        if _onepoint(self.lines, other.lines):
            return True

        if _onepoint(other.lines, self.lines):
            return True

        # check for line collisions
        test_lines = []
        oidx = 0

        # find start of self lines in other lines
        while oidx < other.lines.shape[0] and other.lines[oidx][1][0] < self.lines[0][0][0]:
            oidx += 1

        def _linecoll(oidx):
            for _, l in enumerate(self.lines):
                # add more other lines to the window
                while oidx < other.lines.shape[0] and other.lines[oidx][0][0] <= l[1][0]:
                    heapq.heappush(
                        test_lines, (other.lines[oidx][1][0], other.lines[oidx].tolist())
                    )
                    oidx += 1

                # remove lines that are outside from the window
                while test_lines and test_lines[0][1][1][0] < l[0][0]:
                    heapq.heappop(test_lines)

                for ol in test_lines:
                    if bpy_intersect(l[0], l[1], ol[1][0], ol[1][1]):
                        return True

            return False

        return _linecoll(oidx)

    def _rebuild_lines(self):
        """ Sort segments by first position X and sort individual segments to \
        that point with least X is first """
        b = self.lines

        # sort individual segments
        b = b[np.expand_dims(np.arange(len(self.lines)), axis=1), b[:, :, 0].argsort(), :]

        # sort segments
        b = b[b[:, 0, 0].argsort()]

        self.lines = b

    def _bb_calc(self):
        self.bounding_box.refresh(self.lines[:, :, 0], self.lines[:, :, 1])

    def _fallback(self):
        self.broken = True

        # Use bounding box as collision edges
        pt = self.verts[0][0]
        minx, miny, maxx, maxy = pt[0], pt[1], pt[0], pt[1]

        for i in range(len(self.verts)):
            for j in range(len(self.verts[i])):
                x, y = self.verts[i][j]
                if x > maxx:
                    maxx = x
                if x < minx:
                    minx = x
                if y > maxy:
                    maxy = y
                if y < miny:
                    miny = y

        # print("Reverted island to bb: len:{} minx:{:.2f} maxx:{:.2f} miny:{:.2f}
        # maxy:{:.2f}".format(len(self.verts), minx, maxx, miny, maxy))

        make_it_bb = False

        # if island size below certain threshold, always make into bounding box
        if maxx - minx < 1 / 10000 or maxy - miny < 1 / 10000:
            make_it_bb = True

        for i in self.verts:
            if len(i) < 3:
                make_it_bb = True

        maxx += self.margin
        maxy += self.margin
        minx -= self.margin
        miny -= self.margin

        if len(self.verts) > 4 and not make_it_bb:
            # Convex hull
            ch = cgeo.convex_hull([(j[0], j[1]) for i in self.verts for j in i])
            tlines = [(ch[i], ch[(i + 1) % len(ch)]) for i in range(len(ch))]
            self.chains = [cgeo.Chain(cgeo.chain_connect(tlines))]
            if not cgeo.clockwise(self.chains[0]):
                self.chains[0].reverse()
            # make_it_bb = not self.chains[0].check()

            self.lines = np.empty(shape=(len(tlines), 2, 2), dtype=np.float)
            for i, l in enumerate(tlines):
                self.lines[i] = ((l[0][0], l[0][1]), (l[1][0], l[1][1]))
            print("Rebuilt island as a convex hull.")

        else:
            if maxx - minx < 1 / 10000:
                maxx = minx + 1 / 10000
            if maxy - miny < 1 / 10000:
                maxy = miny + 1 / 10000

            # Bounding box
            self.lines = np.empty(shape=(4, 2, 2), dtype=np.float)
            self.lines[0] = ((minx, maxy), (minx, miny))
            self.lines[1] = ((minx, miny), (maxx, miny))
            self.lines[2] = ((minx, maxy), (maxx, maxy))
            self.lines[3] = ((maxx, miny), (maxx, maxy))

            sl = self.lines
            # self.chains = [cgeo.Chain([sl[0], sl[1], sl[3], sl[2][::-1]])]
            # clockwise
            # self.chains = [cgeo.Chain([sl[2], sl[3][::-1], sl[1][::-1], sl[0][::-1]])]
            self.chains = [cgeo.Chain([sl[0], sl[1], sl[3], sl[2][::-1]])]
            print("Rebuilt island as a bounding box.")

        self._bb_calc()
        self._trs_lines(self.trs_matrix)
        self.uv_area = self.bounding_box.area

        assert self.chains[0].check()
        if not cgeo.clockwise(self.chains[0]):
            self.chains[0].reverse()

    def _trs_lines(self, mtx):
        newlines = np.empty(self.lines.shape)

        x0, y0 = self.lines[:, 0, 0], self.lines[:, 0, 1]
        newlines[:, 0, 0] = mtx[0, 0] * x0 + mtx[0, 1] * y0 + mtx[0, 2]
        newlines[:, 0, 1] = mtx[1, 0] * x0 + mtx[1, 1] * y0 + mtx[1, 2]

        x1, y1 = self.lines[:, 1, 0], self.lines[:, 1, 1]
        newlines[:, 1, 0] = mtx[0, 0] * x1 + mtx[0, 1] * y1 + mtx[0, 2]
        newlines[:, 1, 1] = mtx[1, 0] * x1 + mtx[1, 1] * y1 + mtx[1, 2]

        self.lines = newlines

    def rotate_mar(self):
        """ Rotate to minimum bounding based on convex hull edges """
        ch = cgeo.convex_hull([(j[0], j[1]) for i in self.verts for j in i])
        tlines = [(ch[i], ch[(i + 1) % len(ch)]) for i in range(len(ch))]

        points = np.array(ch)
        min_mar = None
        rot_angle = 0.0
        for tl in tlines:
            v0 = np.array([1.0, 0.0])
            v1 = np.array([tl[1][0] - tl[0][0], tl[1][1] - tl[0][1]])

            ang = cgeo.vec_angle(v0, v1)
            c, s = np.cos(ang), np.sin(ang)
            R = np.matrix([[c, -s], [s, c]])
            t = np.dot(points, R)

            mar = (np.max(t[:, 0]) - np.min(t[:, 0])) * (np.max(t[:, 1]) - np.min(t[:, 1]))
            if min_mar is None or mar < min_mar:
                min_mar = mar
                rot_angle = ang

        if rot_angle != 0.0:
            self.rotate(np.degrees(rot_angle))

    def transform(self, mtx):
        self._trs_lines(mtx)
        self.trs_matrix = np.dot(mtx, self.trs_matrix)

    def move(self, x, y):
        self.bounding_box.move(x, y)

        self.lines[:, :, 0] = self.lines[:, :, 0] + x
        self.lines[:, :, 1] = self.lines[:, :, 1] + y

        mtx_t = np.identity(3)
        mtx_t[0, 2] = x
        mtx_t[1, 2] = y
        self.trs_matrix = np.dot(mtx_t, self.trs_matrix)

    def rotate(self, amount):
        mtx_t = np.identity(3)
        mtx_t[0, 2] = -self.bounding_box.center_x
        mtx_t[1, 2] = -self.bounding_box.center_y
        self.transform(mtx_t)

        theta = np.radians(amount)
        c, s = np.cos(theta), np.sin(theta)
        mtx_r = np.identity(3)
        mtx_r[0, 0] = c
        mtx_r[0, 1] = s
        mtx_r[1, 0] = -s
        mtx_r[1, 1] = c
        self.transform(mtx_r)

        mtx_t = np.identity(3)
        mtx_t[0, 2] = self.bounding_box.center_x
        mtx_t[1, 2] = self.bounding_box.center_y
        self.transform(mtx_t)

        self._rebuild_lines()
        self._bb_calc()

    def rotate_around(self, theta, cx, cy):
        mtx_t = np.identity(3)
        mtx_t[0, 2] = -cx
        mtx_t[1, 2] = -cy
        self.transform(mtx_t)

        c, s = np.cos(theta), np.sin(theta)
        mtx_r = np.identity(3)
        mtx_r[0, 0] = c
        mtx_r[0, 1] = s
        mtx_r[1, 0] = -s
        mtx_r[1, 1] = c
        self.transform(mtx_r)

        mtx_t = np.identity(3)
        mtx_t[0, 2] = cx
        mtx_t[1, 2] = cy
        self.transform(mtx_t)

        self._rebuild_lines()
        self._bb_calc()

    def flipx(self):
        self.transform(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float))
        self._rebuild_lines()
        self._bb_calc()

    def scale_from(self, scale):
        self.transform(np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float))
        self._bb_calc()
        self.uv_area *= scale

    def scale_xy(self, x, y):
        self.transform(np.array([[x, 0, 0], [0, y, 0], [0, 0, 1]], dtype=np.float))
        self._bb_calc()

    def scale_inplace(self, scale):
        self.transform(
            np.array(
                [
                    [1, 0, -self.bounding_box.center_x],
                    [0, 1, -self.bounding_box.center_y],
                    [0, 0, 1],
                ],
                dtype=np.float,
            )
        )
        self.transform(np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float))
        self.transform(
            np.array(
                [[1, 0, self.bounding_box.center_x], [0, 1, self.bounding_box.center_y], [0, 0, 1]],
                dtype=np.float,
            )
        )
        self._bb_calc()
        self.uv_area *= scale

    def set_location(self, x, y):
        dispx = x - self.bounding_box.min_x
        dispy = y - self.bounding_box.min_y
        self.move(dispx, dispy)

    def update_verts(self, mtx):
        for i in range(len(self.verts)):
            for j in range(len(self.verts[i])):
                x, y = self.verts[i][j]
                self.verts[i][j] = (np.dot(mtx, [x, y, 1])[:2]).tolist()
