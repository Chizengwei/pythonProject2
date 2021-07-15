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


import copy
import math
import time
import random
from collections import deque, defaultdict

import numpy as np

import bpy
from mathutils import geometry as gm

rnd = random.random
bpy_intersect = gm.intersect_line_line_2d


class Packing:
    def __init__(self, uv_hnd):
        self._start_0 = time.time()

        print("Generate manipulation set.")
        self.uv_hnd = uv_hnd
        self.isles, self.groups = self.uv_hnd.generate_isle_set(
            bpy.context.scene.s_packing.margin / 100.0
        )

        self.iters = bpy.context.scene.s_packing.iterations
        self.rotstep = int(bpy.context.scene.s_packing.rotations)
        self.flip_isle = bpy.context.scene.s_packing.flip
        self.packing_behavior = bpy.context.scene.s_packing.selectbehavior
        self.init_with_mar = bpy.context.scene.s_packing.init_with_mar
        self.iterate = bpy.context.scene.s_packing.iterate

        if uv_hnd.texture_ratio < 1:
            self.uv_x = 1.0
            self.uv_y = 1.0 / uv_hnd.texture_ratio
        else:
            self.uv_x = 1.0 * uv_hnd.texture_ratio
            self.uv_y = 1.0

        # the current bounding box for placing islands
        self.x_limit = self.uv_x
        self.y_limit = self.uv_y

        self.shuffle_max_depth = 5
        self.shuffle_depth = 1
        self.shuffle_index = 0

        self.shuffle_found = False
        self.shuffle_hg_depth = defaultdict(int)
        self.shuffle_hg_index = defaultdict(int)

        self.shuffle_solutions = 0
        self.shuffle_iterations = 0

        self.best_mtx_list = []
        self.best_grouping = copy.deepcopy(self.groups)
        self.best_uv_area = 0.0
        self.best_rotations = deque([0 for _ in range(len(self.isles))])
        self.save_if_better()

        from . import accelerator  # , tools
        self.packer = accelerator.Packer()

        print("Packing init in {:.3f} seconds".format(time.time() - self._start_0))

        self._start()

    def _start(self):
        self.packer.new(self.iters)

        # start iteration
        self.packable_islands = [i[0] for i in self.groups][::-1]
        print("Packed islands:", len(self.packable_islands))

        # pack: selected
        # pack selected and visible islands into unselected and visible, scaling to fit
        #
        # pack: lock
        # pack selected and visible islands into unselected and visible, maintaining scale
        #
        # pack: ignore
        # ignore unselected and hidden islands completely, pack only visible and selected
        #
        # pack: all
        # pack all islands, including hidden and unselected, as long as they are part of the UV

        # determine which UV islands are selected and which are not
        # assumption: if any in group is selected, all are selected
        unpacked_groups = []
        for n, i in enumerate(self.groups):
            if not self.isles[i[0]].selected:
                unpacked_groups.append(n)

        self.unselected_islands = []
        for i in unpacked_groups:
            self.unselected_islands.append(self.groups[i][0])

        print("Selected groups:", len(self.unselected_islands))

        # if ignore islands option selected, completely ignore unselected islands
        if self.packing_behavior == "ignore":
            for i in self.unselected_islands:
                if i in self.packable_islands:
                    self.packable_islands.remove(i)

        # if setting is not "all", remove all hidden islands
        # assumption: if any in group is visible, all are visible
        if self.packing_behavior != "all":
            self.packable_islands = [i for i in self.packable_islands if not self.isles[i].hidden]
            self.unselected_islands = [
                i for i in self.unselected_islands if not self.isles[i].hidden
            ]

        self.packed_islands = []

        self.uv_area = sum([self.isles[i].uv_area for i in self.packable_islands])
        print("UV area:", self.uv_area)

        # scale all islands that are included in the processing to max
        if self.packing_behavior != "lock":
            # prescale islands to a good size
            if self.uv_area > 0:
                amount = 1.0 / (self.uv_area * 2.0)
            else:
                amount = 1.0

            if amount < 2:
                # only scale down
                for isle_id in self.packable_islands:
                    # Rely on smart scaling based on island bounding boxes
                    # instead of initial UV scale
                    self.isles[isle_id].scale_from(amount)

        # ...else, if pack only selected chosen, write unselected islands without packing
        if self.packing_behavior == "selected" or self.packing_behavior == "lock":
            # write unselected without packing (but do maintain scale and collision test on them)
            for i in self.unselected_islands:
                if i in self.packable_islands:
                    self.place(i)
                    self.packer.add(self.isles[i])
                    self.packable_islands.remove(i)

        # continue
        self.zero_packable = True

        if self.packable_islands:
            # start packing
            # scale all islands to a good size

            # set packing area to the widest/tallest island bounding box
            start_x = max([self.isles[i].bounding_box.width for i in self.packable_islands])
            start_y = max([self.isles[i].bounding_box.height for i in self.packable_islands])

            if len(self.unselected_islands) > 0 and self.packing_behavior != "ignore":
                start_x = max([self.isles[i].bounding_box.max_x for i in self.unselected_islands])
                start_y = max([self.isles[i].bounding_box.max_y for i in self.unselected_islands])

            if start_x > 1.0:
                start_x = 1.0

            if start_y > 1.0:
                start_y = 1.0

            if start_x > start_y:
                self._set_new_x(start_x)
            else:
                self._set_new_y(start_y)

            self.zero_packable = False
            self.current_island = self.packable_islands.pop()
            self.iter_count = 0

            print("New iteration at {:.3f} seconds".format(time.time() - self._start_0))

            if not self.iterate:
                # sort island variations according to width (thinnest first)
                pass
            else:
                # start with random rotation
                self.current_rotations = list(self.best_rotations)

    def place(self, island_id):
        if math.isnan(self.isles[island_id].bounding_box.width) or math.isnan(
            self.isles[island_id].bounding_box.height
        ):
            self.isles[island_id]._fallback()

        # self.xbuckets.add(self.isles[island_id])
        self.packed_islands.append(island_id)

    def save_if_better(self):
        if self.is_uv_better():
            self.save_uv()

    def is_uv_better(self):
        self.uv_area = sum([self.isles[i[0]].uv_area for i in self.groups])
        # print("Achieved UV area: ", self.uv_area)

        return self.uv_area > self.best_uv_area

    def save_uv(self):
        print()
        print("Saving new UV solution...")
        print()

        # save solution
        self.best_grouping = copy.deepcopy(self.groups)
        self.best_mtx_list = []

        for group in self.groups:
            self.best_mtx_list.append(self.isles[group[0]].trs_matrix)

        self.best_uv_area = self.uv_area

    def _pack_end(self):
        # final scale back to 1.0 range
        if self.packing_behavior != "lock":
            amount = self.uv_x / self.x_limit
            print("adjust:", amount)
            for isle_id in self.packed_islands:
                self.isles[isle_id].scale_from(amount)

        if self.iterate:
            if self.is_uv_better():
                self.shuffle_found = True
                self.shuffle_solutions += 1
                self.shuffle_hg_depth[self.shuffle_depth] += 1
                self.shuffle_hg_index[self.shuffle_index] += 1
                self.save_uv()

            total_area = 0.0
            self.shuffle_iterations += 1

            # find a block with good enough size
            # while total_area < 0.01/self.best_uv_area:
            while total_area < 0.001 / self.best_uv_area:
                self.groups = copy.deepcopy(self.best_grouping)

                if len(self.groups) // (2 ** self.shuffle_depth) < 2:
                    self.shuffle_depth = 1

                    # if scan found 0 new solutions, exit
                    if self.shuffle_found is False:
                        return False

                    self.shuffle_found = False

                blen = len(self.groups) // (2 ** self.shuffle_depth)

                print(
                    "Shuffling...",
                    self.shuffle_depth,
                    self.shuffle_index,
                    "(",
                    self.shuffle_solutions,
                    "/",
                    self.shuffle_iterations,
                    ")=",
                    self.shuffle_solutions / self.shuffle_iterations,
                )

                print(self.shuffle_hg_depth)
                print(self.shuffle_hg_index)

                print("blen:", blen, self.shuffle_index, self.shuffle_depth)
                print("<", self.shuffle_index * blen, self.shuffle_index * blen + blen - 1, ">")

                shuffled = []
                for i in range(blen // 2):
                    ia = self.shuffle_index * blen + i
                    ib = self.shuffle_index * blen + blen // 2 + i

                    shuffled.append(self.groups[ia][0])
                    shuffled.append(self.groups[ib][0])

                    a = self.groups[ia]
                    self.groups[ia] = self.groups[ib]
                    self.groups[ib] = a

                total_area = 0.0
                for sh in shuffled:
                    total_area += self.isles[sh].uv_area

                self.shuffle_index += 1

                # refactor into changing quicksort style regions instead of just one piece
                print(self.shuffle_index, 2 ** (self.shuffle_depth - 1))
                if self.shuffle_index >= 2 ** (self.shuffle_depth - 1):
                    self.shuffle_index = 0
                    self.shuffle_depth += 1

            self._start()

            return True

        else:
            # close packer process
            # if self.accelerator:
            #    self.packer.end()

            return False

    def pack(self):
        if self.zero_packable:
            return False

        if self.iterate:
            bpy.context.scene.s_packing.uv_packing_progress = "{}/{}{}".format(
                self.shuffle_solutions, self.shuffle_iterations, "." * self.iter_count
            )
        else:
            bpy.context.scene.s_packing.uv_packing_progress = "{:.2f}%{}".format(
                len(self.packed_islands) * 100.0 / len(self.isles), "." * self.iter_count
            )

        if math.isnan(self.isles[self.current_island].bounding_box.width) or math.isnan(
            self.isles[self.current_island].bounding_box.height
        ):
            print("Error: NaN")
            print(len(self.isles), self.current_island)
            self.isles[self.current_island]._fallback()
        else:
            this_isle = self.isles[self.current_island]
            if not this_isle.picked:
                # if self.packing_behavior != 'lock':
                #     this_isle.scale_from(self.current_scale)
                this_isle.picked = True

            if self.run(self.current_island):
                self.place(self.current_island)
                self.iter_count = 0

                if self.packable_islands:
                    self.current_island = self.packable_islands.pop()
                else:
                    # ---- END OF PACKING
                    return self._pack_end()

            else:
                self.iter_count = (self.iter_count + 1) % 4

        return True

    def _set_new_x(self, new_x):
        scale_factor = new_x / self.x_limit

        self.x_limit *= scale_factor
        self.y_limit *= scale_factor

        # print("Scale:", scale_factor, ", X limit:", self.x_limit, ", Y limit:", self.y_limit)
        # print("Scale X: {}, X limit: {}, Y limit: {}".
        # format(scale_factor, self.x_limit, self.y_limit))

    def _set_new_y(self, new_y):
        scale_factor = new_y / self.y_limit

        self.x_limit *= scale_factor
        self.y_limit *= scale_factor

        # print("Scale:", scale_factor, ", X limit:", self.x_limit, ", Y limit:", self.y_limit)
        # print("Scale Y: {}, X limit: {}, Y limit: {}".format(scale_factor,
        # self.x_limit, self.y_limit))

    def _adjust_new_x(self, new_x):
        if new_x > self.x_limit:

            scale_factor = new_x / self.x_limit
            # print("newx", scale_factor)
            # print("Adjust X:", scale_factor)

            self.x_limit *= scale_factor
            self.y_limit *= scale_factor

    def _adjust_new_y(self, new_y):
        if new_y > self.y_limit:
            scale_factor = new_y / self.y_limit
            # print("newy", scale_factor)
            # print("Adjust Y:", scale_factor)

            self.x_limit *= scale_factor
            self.y_limit *= scale_factor

    def cancel(self):
        # if iterating better solution, write the best current one
        if self.iterate and self.best_mtx_list != []:
            print("Writing best result...")
            self.write_saved()

    def write_saved(self):
        """ Write previously saved state data to UV """

        # single group shares a single transformation matrix
        if hasattr(self, "zero_packable") and not self.zero_packable:
            for i, group in enumerate(self.best_grouping):
                mtx = self.best_mtx_list[i]
                isle_id = group[0]
                for _, isle_id in enumerate(group):
                    isle = self.isles[isle_id]
                    isle.update_verts(mtx)
                    self.uv_hnd.write(isle)

    def write(self):
        """ Write island data to UV """

        # single group shares a single transformation matrix
        if hasattr(self, "zero_packable") and not self.zero_packable:
            for group in self.groups:
                mtx = self.isles[group[0]].trs_matrix
                isle_id = group[0]
                for _, isle_id in enumerate(group):
                    isle = self.isles[isle_id]
                    isle.update_verts(mtx)
                    self.uv_hnd.write(isle)

    def run(self, p):
        # find a free location
        thisisle = self.isles[p]

        if self.init_with_mar:
            thisisle.rotate_mar()

        # Islands with bounding boxes over 1.0 can't fit so scale accordingly
        # bugfix: 1.0001 instead of 1.0
        self._adjust_new_x(thisisle.bounding_box.width * 1.0001)
        self._adjust_new_y(thisisle.bounding_box.height * 1.0001)

        # x, y, selected id, max x reached
        valid, rx, ry, rotation, mirror = self.packer.find(
            thisisle, 4.0, self.y_limit, rotations=self.rotstep, mirror=self.flip_isle
        )

        thisisle.rotate(np.degrees(-rotation))
        if mirror > 0:
            thisisle.flipx()
        thisisle._bb_calc()
        thisisle.set_location(rx, ry)

        self.isles[p] = thisisle
        self.packer.confirm()

        self._adjust_new_x(thisisle.bounding_box.max_x)
        self._adjust_new_y(thisisle.bounding_box.max_y)

        return True
