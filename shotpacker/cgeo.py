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


import math
import numpy as np
import pyclipper
from collections import defaultdict


def dist_point_line(p, l1, l2):
    x0, y0 = p
    x1, y1 = l1
    x2, y2 = l2
    y21 = y2 - y1
    x21 = x2 - x1
    if l1 == l2:
        return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return abs(y21 * x0 - x21 * y0 + x2 * y1 - y2 * x1) / math.sqrt(y21 ** 2 + x21 ** 2)


def kind_of(a, b, m):
    return (a + m) > b and (a - m) < b


def kind_of_ratio(a, b, m):
    return a * (1 + m) > b and a * (1 - m) < b


def array_kind_of(a, b, m):
    if len(a) != len(b) or len(m) != len(a):
        return False

    for i in range(len(a)):
        if not kind_of(a[i], b[i], m[i]):
            return False

    return True


def array_kind_of_ratio(a, b, m):
    if len(a) != len(b) or len(m) != len(a):
        return False

    for i in range(len(a)):
        if not kind_of_ratio(a[i], b[i], m[i]):
            return False

    return True


def norm_vec(x, y):
    v = np.array([x, y])
    l = np.linalg.norm(v)
    return v / l


def vec_angle(v0, v1):
    return np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))


def convex_hull(vertices):
    """ Computes the convex hull of a set of 2D points.
        Andrew's monotone chain algorithm. O(n log n) complexity. """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(vertices))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]


def clockwise(ch):
    s = 0
    for e in ch:
        s += (e[1][0] - e[0][0]) * (e[1][1] + e[0][1])
    return s > 0


class Chain:
    __slots__ = ['chain']

    def __init__(self, chain):
        self.chain = chain
        assert len(self.chain) > 2
        # TODO:
        # assert self.check()

    def __len__(self):
        return len(self.chain)

    def __repr__(self):
        res = []
        for i in self.chain:
            res.append(
                "({:.2f}, {:.2f} => {:.2f}, {:.2f})".format(i[0][0], i[0][1], i[1][0], i[1][1])
            )
        return "[" + ", ".join(res) + "]"

    def __getitem__(self, index):
        return self.chain[index]

    def length(self):
        total_len = 0.0
        for a0, b0 in self.chain:
            total_len += np.linalg.norm(np.array([b0[0] - a0[0], b0[1] - a0[1]]))
        return total_len

    def longest_diagonal(self):
        dg_len = 0.0
        initial = self.chain[0][0]
        best_index = 0
        for i, (a, _) in enumerate(self.chain):
            dg = np.linalg.norm(np.array([a[0] - initial[0], a[1] - initial[1]]))
            if dg > dg_len:
                dg_len = dg
                best_index = i
        return best_index

    def bounding_box(self):
        maxx, maxy = self.chain[0][0]
        minx, miny = self.chain[0][0]
        for a, b in self.chain:
            if a[0] > maxx:
                maxx = a[0]
            if a[1] > maxy:
                maxy = a[1]
            if a[0] < minx:
                minx = a[0]
            if a[1] < miny:
                miny = a[1]

        return minx, miny, maxx, maxy

    def check(self):
        """ Check that chain is looped and doesn't have zero length segments """
        l = self.chain
        e = set()
        for i, il in enumerate(l):
            tp = (*(l[(i + 1) % len(l)][0]),)
            il1 = (*il[1],)
            il0 = (*il[0],)

            if (il0, il1) not in e:
                e.add((il0, il1))
            else:
                return False

            if il1 != tp or il1 == il0:
                # print(il0, il1, tp)
                return False

        return True

    def dot_string(self):
        dots_a = []

        total_len = self.length()

        for i in range(len(self.chain)):
            a0, b0 = self.chain[i]
            a1, b1 = self.chain[(i + 1) % len(self.chain)]

            x0, y0 = b0[0] - a0[0], b0[1] - a0[1]
            v0 = norm_vec(x0, y0)

            x1, y1 = b1[0] - a1[0], b1[1] - a1[1]
            v1 = norm_vec(x1, y1)

            plen = np.linalg.norm(np.array([x0, y0])) / total_len
            # dots_a.append((1.0, plen))
            dots_a.append((v0.dot(v1), plen))

        # dot product of edge and the next edge, edge length
        return dots_a

    def norms(self, distance):
        ch = self.chain
        lrot = []
        for (a, b) in ch:
            n = b[0] - a[0], b[1] - a[1]
            # 90 degree rotation
            n = np.array([-n[1], n[0]])
            x = np.linalg.norm(n)
            if x == 0:
                x = 0.0000001
            n = n / x
            lrot.append(n)

        lnorms = []
        for i, l in enumerate(lrot):
            norm = lrot[i]
            # FIXME: divide by zero might cause NaN, why?
            x = np.linalg.norm(norm)
            if x == 0:
                x = 0.0000001
            norm = -(norm * distance) / x
            lnorms.append(norm)
        return lnorms

    def prune(self, margin):
        ch = self.chain
        start_link = 0
        current_link = start_link + 2
        new_chain = []
        while start_link < len(ch) - 2 and current_link < len(ch):
            in_bound = True
            for i in range(start_link + 1, current_link):
                dist = dist_point_line(ch[i][0], ch[start_link][0], ch[current_link][0])
                if dist > margin:
                    in_bound = False
                    break

            if in_bound:
                current_link += 1
            else:
                new_chain.append((ch[start_link][0], ch[current_link - 1][0]))
                start_link = current_link - 1
                current_link = start_link + 2

        for i in range(start_link, len(ch)):
            new_chain.append((ch[i][0], ch[i][1]))

        self.chain = new_chain
        return self

    def douglas_peucker(self, margin):
        ch = self.chain

        def _dpcall(pl, eps):
            dmax = 0
            index = 0
            end = len(pl)
            if end == 1:
                return (pl[0],)

            for i in range(1, end - 1):
                d = dist_point_line(pl[i], pl[0], pl[end - 1])
                if d > dmax:
                    index = i
                    dmax = d

            if dmax <= eps:
                return (pl[0],) + (pl[end - 1],)
            else:
                return _dpcall(pl[0:index], eps) + _dpcall(pl[index:end], eps)

        # def _dpcall3(chain, eps):
        #     stack = [(0, len(chain) - 1)]
        #     result = []
        #     while stack:
        #         pl0, pl1 = stack.pop()

        #         dmax = 0
        #         index = 0

        #         x1, y1 = chain[pl0]
        #         x2, y2 = chain[pl1]
        #         y21 = y2 - y1
        #         x21 = x2 - x1
        #         tsq2 = math.sqrt(y21 ** 2 + x21 ** 2)
        #         xyyx = x2 * y1 - y2 * x1
        #         for i in range(pl0 + 1, pl1):
        #             x0, y0 = chain[i]
        #             d = 0.0

        #             if x1 == x2 and y1 == y2:
        #                 d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        #             else:
        #                 d = abs(y21 * x0 - x21 * y0 + xyyx) / tsq2

        #             if d > dmax:
        #                 index = i
        #                 dmax = d

        #         if dmax <= eps:
        #             result.append(chain[pl0])
        #         else:
        #             stack.append((index, pl1))
        #             stack.append((pl0, index - 1))
            
        #     return result

        res = _dpcall([i[0] for i in ch], margin)
        if len(res) > 2:
            self.chain = list(zip(res, res[1:])) + [(res[-1], res[0])]
        return self

    def reverse(self):
        res = []
        for i in self.chain[::-1]:
            res.append(i[::-1])
        self.chain = res
        return self

    def mirror(self):
        res = []
        for i in self.chain:
            res.append(((-i[0][0], i[0][1]), (-i[1][0], i[1][1])))
        self.chain = res
        return self

    def clockwise(self):
        return clockwise(self.chain)

    def dilate(self, distance):
        """ Dilate an ordered edge loop high quality """
        if clockwise(self.chain):
            self.reverse()
        # assert not clockwise(self.chain)
        ch = self.chain
        lnorms = self.norms(distance)

        # print("normal dilate:",len(ch))

        # move lines to create dilation
        new_chain = []
        for j, val in enumerate(ch):
            x0, y0 = lnorms[j]
            x1, y1 = lnorms[(j + 1) % len(lnorms)]

            m_vec = lnorms[j] + lnorms[(j + 1) % len(lnorms)]
            x = np.linalg.norm(m_vec)
            if x == 0:
                x = 0.0000001
            cnx, cny = m_vec * distance / x

            nx, ny = ch[(j + 1) % len(ch)][0][0], ch[(j + 1) % len(ch)][0][1]
            epx, epy = ch[j][1][0], ch[j][1][1]

            new_chain.append(((ch[j][0][0] + x0, ch[j][0][1] + y0), (epx + x0, epy + y0)))
            new_chain.append(((epx + x0, epy + y0), (epx + cnx, epy + cny)))
            new_chain.append(((epx + cnx, epy + cny), (nx + x1, ny + y1)))

        # print('---')
        # print(chains[chi], len(chains[chi]))
        # print(new_chain, len(new_chain))
        self.chain = new_chain
        return self


def chain_connect(l):
    """ Makes a chain out of a list of 2D segments if possible """
    # TODO: doesn't create multiple chains, creates only one (if possible) and quits

    def _find_connection(s):
        for i in l:
            if i != s:
                if i[0] == s[1]:
                    return i
                if i[1] == s[1]:
                    return i[::-1]
        return None

    chain = []
    latest = l[0]
    while len(chain) < len(l):
        s = _find_connection(latest)
        if s is not None:
            chain.append(s)
            latest = s
        else:
            break

    return chain


def eat_chain(ch, ln):
    # <a> with same start-point as last segment end-point
    l = [a for a in ln if ch[-1][-1][1] == a[0] and a not in ch[-1]]
    if l:
        ln.remove(l[0])
        return l[0]
    else:
        return None


def find_mesh_chains(verts):
    """ Find 2d mesh rim edges from face vertex loops (outer and inner) """
    # TODO: based on "nonmanifold" edges and building loops from that instead of ... this
    # find rim edges
    edges = defaultdict(int)
    dir = defaultdict(bool)

    for i in range(len(verts)):
        vertloop = tuple(tuple(j) for j in verts[i])
        iloop = [
            (vertloop[j], vertloop[(j + 1) % len(vertloop)]) for j in range(0, len(vertloop), 1)
        ]

        cwise = clockwise(iloop)

        # enforce face rotation direction
        if cwise:
            iloop.reverse()
            iloop = [(j[1], j[0]) for j in iloop]

        # for every edge in face...
        for (a, b) in iloop:
            d = a > b
            if a > b:  # if a is greater, swap a and b
                a, b = b, a
            edges[(a, b)] += 1
            dir[(a, b)] = d

    edges = {k: v for k, v in edges.items() if v == 1}
    rimlines = [k for k in edges.keys()]

    # generate chains
    # rebuild clockwise edges
    mlines = []
    for (a, b) in rimlines:
        if dir[(a, b)]:
            a, b = b, a
        mlines.append((a, b))

    chains = []

    if len(mlines) > 0:
        chains.append([mlines[0]])
        mlines.remove(mlines[0])
    else:
        # print("Constructing island failed: no chains found.")
        return False

    while mlines:
        i = eat_chain(chains, mlines)
        if not i:
            chains.append([mlines[0]])
            mlines.remove(mlines[0])
        else:
            chains[-1].append(i)

    for c in chains:
        invalid = False
        for i, s in enumerate(c):
            invalid |= c[i][1][0] != c[(i + 1) % len(c)][0][0]
            invalid |= c[i][1][1] != c[(i + 1) % len(c)][0][1]
        # if not (len(c) > 1 or c[-1][1] == c[0][0]):
        if len(c) < 2 or invalid:
            print("Warning: island has invalid chains.")
            return None

    return [Chain(c) for c in chains]


def poly_clockwise(poly):
    s = 0
    for i, e in enumerate(poly):
        ne = poly[(i + 1) % len(poly)]
        s += (ne[0] - e[0]) * (ne[1] + e[1])
    return s > 0


def poly_area(poly):
    try:
        if len(poly) > 2:
            # Shoelace formula: 1/2 (sum(Xi*Yi+1) + XnY1 - sum(Xi+1Yi) - X1Yn)
            a = 0.0
            p_len = len(poly)
            for i in range(p_len):
                # if np.isnan(poly[i][0]):
                #     print("NAN NAN NAN NAN NAN NAN NAN NAN NAN ")
                #     print(poly)

                a += poly[i][0] * poly[(i + 1) % p_len][1]
                a -= poly[(i + 1) % p_len][0] * poly[i][1]
            a /= 2.0
            return abs(a)
    except TypeError as te:
        # print(te)
        print(poly)
        raise TypeError(te)

    return 0.0


def triangle_area(self, verts):
    # Heron's formula
    a = (verts[1][0] - verts[0][0]) ** 2.0 + (verts[1][1] - verts[0][1]) ** 2.0
    b = (verts[2][0] - verts[1][0]) ** 2.0 + (verts[2][1] - verts[1][1]) ** 2.0
    c = (verts[0][0] - verts[2][0]) ** 2.0 + (verts[0][1] - verts[2][1]) ** 2.0

    cal = (2 * a * b + 2 * b * c + 2 * c * a - a ** 2 - b ** 2 - c ** 2) / 16
    if cal < 0:
        cal = 0
    return math.sqrt(cal)


def quad_area(self, verts):
    return self.triangle_area(verts[:3]) + self.triangle_area(verts[2:] + [verts[0]])


def clipper_offset(chains, margin):
    # TODO: Bug in which vert-connection (not edge) doesn't correctly
    # generate inner chain (polygon hole)
    if margin < 0.0000:
        margin = 0.0000

    clipper_offset = pyclipper.PyclipperOffset()

    scale = 20000.0

    # print("\nC0:",end='')

    for ci, c in enumerate(chains):
        # set correct direction for paths as an input for Pyclipper
        if ci == 0 and not clockwise(c):
            # print("PyClipper WARNING! Invalid chain[0] direction")
            chains[ci].reverse()

        if ci > 0 and clockwise(c):
            # print("PyClipper WARNING! Invalid chain[1+] direction")
            chains[ci].reverse()

        # print(">" if clockwise(c) else "<", end='')

        coord = [(int(i[0][0] * scale), int(i[0][1] * scale)) for i in c]
        clipper_offset.AddPath(coord, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)

    new_coordinates = clipper_offset.Execute(int(margin * scale))
    ret_chains = []
    # print("\nC1:",end='')
    for ci, c in enumerate(new_coordinates):
        new_c = []
        for i, it in enumerate(c):
            nit = c[(i + 1) % len(c)]
            new_c.append(((it[0] / scale, it[1] / scale), (nit[0] / scale, nit[1] / scale)))
        ret_chains.append(Chain(new_c))
        # print(">" if clockwise(ret_chains[-1]) else "<", end='')

        assert ret_chains[-1].check()

    return ret_chains


def clipper_bb_normalize(poly, always=False):
    # ensure chain bounding box isn't too small
    xt, yt = poly[0][0], poly[0][1]
    minx, miny, maxx, maxy = xt, yt, xt, yt
    for i in poly:
        x, y = i[0], i[1]
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
        if x < minx:
            minx = x
        if y < miny:
            miny = y

    if maxx - minx < 1 or maxy - miny < 1 or always:
        print("!!! CRITICAL !!!: Invalid chain in clipper_island_union")
        poly = [(), (), (), ()]
        maxx += 1 * (maxx == minx)
        maxy += 1 * (maxy == miny)
        poly[3] = (minx, miny)
        poly[2] = (maxx, miny)
        poly[1] = (maxx, maxy)
        poly[0] = (minx, maxy)

    return poly


def clipper_island_union(isles):
    # print("  init pyclipper")
    pc = pyclipper.Pyclipper()
    scale = 20000.0

    for ii, isle in enumerate(isles):
        # print("  >> next isle")
        for ci, c in enumerate(isle.chains):
            poly = []
            for i in c:
                x, y = int(i[0][0] * scale), int(i[0][1] * scale)
                poly.append((x, y))

            poly = clipper_bb_normalize(poly)

            # TODO: save clockwise before float -> int conversion
            if ci == 0 and not poly_clockwise(poly):
                poly = poly[::-1]

            if ci > 0 and poly_clockwise(poly):
                # inner holes need to be counter clockwise orientation for the algo
                poly = poly[::-1]

            # print("  cw:", cgeo.poly_clockwise(poly))

            try:
                if ii == 0:
                    # print("  addpath subject")
                    pc.AddPath(poly, pyclipper.PT_SUBJECT, True)
                else:
                    # print("  addpath clip")
                    pc.AddPath(poly, pyclipper.PT_CLIP, True)
            except pyclipper.ClipperException as e:
                print("Clipper exception")
                print("-----------------")
                # cgeo.repr_chain(c)
                print("len:", c.length())
                print("c:", c)
                print("poly:", poly)
                print("chain check:", c.check())
                print("broken island (a, b):", isles[0].broken, isles[1].broken)
                print(e)
                # raise pyclipper.ClipperException(e)
                # Try once more, if this fails: DEAD
                poly = clipper_bb_normalize(poly, always=True)
                if ii == 0:
                    # print("  addpath subject")
                    pc.AddPath(poly, pyclipper.PT_SUBJECT, True)
                else:
                    # print("  addpath clip")
                    pc.AddPath(poly, pyclipper.PT_CLIP, True)

    # print("  clipper execute")
    new_coordinates = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    ret_chains = []
    for ci, c in enumerate(new_coordinates):
        new_c = []
        tc = c
        for i, it in enumerate(tc):
            nit = tc[(i + 1) % len(tc)]
            new_c.append(((it[0] / scale, it[1] / scale), (nit[0] / scale, nit[1] / scale)))
        ret_chains.append(Chain(new_c))

    # for i in ret_chains:
    #     print(cgeo.clockwise(i))
    # print("ret")

    return ret_chains
