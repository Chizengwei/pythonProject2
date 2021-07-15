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


class PackerInterface:
    """ Interface for polygon packer """

    def __init__(self):
        raise NotImplementedError("Implement __init__()")

    def new(self, iters, threads):
        raise NotImplementedError("Implement new()")

    def confirm(self):
        raise NotImplementedError("Implement confirm()")

    def add(self, isle):
        raise NotImplementedError("Implement confirm()")

    def find(self, isle, x_limit, y_limit, rotations, mirror):
        raise NotImplementedError("Implement find()")
