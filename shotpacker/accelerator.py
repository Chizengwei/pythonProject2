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


# import ctypes
import sys
from cffi import FFI
import numpy as np
import os.path

# import time

ffilib = None

from .packer_iface import PackerInterface

print("Loading Shotpacker dynamic library FFI")

prefix = {"win32": ""}.get(sys.platform, "lib")
extension = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")

dll_name = "Accelerator" + os.path.sep + prefix + "rs_packer" + extension
path_root = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
dllabspath = path_root + dll_name
print("Library name: ", dllabspath)

# use CFFI Python module to read C-type function descriptions to build Python functions
ffi = FFI()

with open(path_root + "Accelerator" + os.path.sep + "header.h", "r") as input_file:
    cstr = input_file.read()
    cstr = "".join([i for i in cstr.split("\n") if i and i[0] != "#"])
    ffi.cdef(cstr)

ffilib = ffi.dlopen(dllabspath)


class Packer(PackerInterface):
    """ Packer module """

    def __init__(self):
        ffilib.init_segments(1000)

    def new(self, iters):
        ffilib.clear_segments()
        ffilib.init_segments(iters)

    def confirm(self):
        ffilib.confirm_placement()

    def add(self, isle):
        segs = np.array(isle.lines, dtype=np.float64)
        cptr = ffi.cast("double *", ffi.from_buffer(segs))
        ffilib.add_segments(cptr, segs.shape[0])

    def find(self, isle, x_limit, y_limit, rotations, mirror):
        segs = np.array(isle.lines, dtype=np.float64)
        cptr = ffi.cast("double *", ffi.from_buffer(segs))
        res = ffilib.find_segments(cptr, segs.shape[0], rotations, mirror, x_limit, y_limit)
        assert res.valid
        return (res.valid, res.x, res.y, res.rotation, res.mirror)
