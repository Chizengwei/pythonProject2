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

import cProfile
import pstats
import io
import json


def writeIslandsJSON(isles, json_out):
    # output JSON representation of islands/lines
    out = []
    for i in isles:
        out.append([(*a[0], *a[1]) for a in i.lines])
    # parms = []
    # for i in isles:
    #     parms.append([i.minx, i.miny, i.maxx, i.maxy, i.avg_x, i.avg_y])
    with open(json_out, "w") as outfile:
        json.dump([out], outfile)
    print("Saved JSON dump to {}.".format(json_out))


def get_teximage_dims(context):
    teximage = None
    for area in context.screen.areas:
        if area.type == "IMAGE_EDITOR":
            teximage = area.spaces.active.image
            break

    if teximage is not None and teximage.size[1] != 0:
        return (teximage.size[0], teximage.size[1])
    else:
        return (None, None)


def clean_fringe(bm, max_angle):
    # --- cleanup UV fringe
    traverse_points = []
    for _, e in enumerate(bm.edges):
        if e.seam:
            # get sum of connected seams for each vertice in the edge
            c_seams_0 = sum(s.seam for s in e.verts[0].link_edges)
            c_seams_1 = sum(s.seam for s in e.verts[1].link_edges)
            if c_seams_0 < 2 or c_seams_1 < 2:
                # if the edge with the seam doesn't have connecting edges with seam in
                # both verts, start traversing along it
                e.seam = False
                if c_seams_0 > 1:
                    traverse_points.append((e.verts[0], e.index))
                elif c_seams_1 > 1:
                    traverse_points.append((e.verts[1], e.index))

    while traverse_points:
        new_points = []
        for v in traverse_points:
            # get all edges that have seam and are not the originating edge
            f_seams = [e for e in v[0].link_edges if e.seam and e.index != v[1]]
            if len(f_seams) == 1:
                # only one new connecting seam found, follow it
                e = f_seams[0]

                # test for angle sharpness
                if e.calc_face_angle(None) < max_angle:
                    e.seam = False
                    new_vert = e.verts[0] if e.verts[0].index != v[0].index else e.verts[1]
                    new_points.append((new_vert, e.index))

        traverse_points = new_points[:]


def profiling_start():
    # profiling
    pr = cProfile.Profile()
    pr.enable()
    return pr


def profiling_end(pr):
    # end profile, print results
    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats(sortby).print_stats(20)
    print(s.getvalue())
