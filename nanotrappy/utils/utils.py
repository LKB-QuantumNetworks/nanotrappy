import numpy as np
import sys, os


def set_axis_index(axis):
    """Set index of axis in the XYZ basis

    Args:
        axis (str): chose axis, among X, Y and Z

    Returns:
        int: index of the chosen axis in the XYZ basis
    """
    if axis.upper() == "X":
        return 0
    elif axis.upper() == "Y":
        return 1
    elif axis.upper() == "Z":
        return 2
    else:
        return ValueError("This axis does not exist.")


def set_normal_axis(plane):
    if plane.upper() == "XY":
        axis = "Z"
    elif plane.upper() == "YZ":
        axis = "X"
    elif plane.upper() == "XZ":
        axis = "Y"
    return axis


def set_axis_from_plane(plane, obj):
    if plane.upper() == "XY":
        coord1 = obj.x
        coord2 = obj.y
    elif plane.upper() == "YZ":
        coord1 = obj.y
        coord2 = obj.z
    elif plane.upper() == "XZ":
        coord1 = obj.x
        coord2 = obj.z
    return coord1, coord2


def set_axis_index_from_plane(plane):
    if plane.upper() == "XY":
        coord1 = 0
        coord2 = 1
        coord3 = 2
    elif plane.upper() == "YZ":
        coord1 = 1
        coord2 = 2
        coord3 = 0
    elif plane.upper() == "XZ":
        coord1 = 0
        coord2 = 2
        coord3 = 1
    return coord1, coord2, coord3


def set_axis_index_from_axis(axis):
    if axis.upper() == "X":
        coord1 = 0
        coord2 = 1
        coord3 = 2
    elif axis.upper() == "Y":
        coord1 = 1
        coord2 = 0
        coord3 = 2
    elif axis.upper() == "Z":
        coord1 = 2
        coord2 = 0
        coord3 = 1
    return coord1, coord2, coord3


def set_axis_from_axis(axis, obj):
    if axis.upper() == "X":
        obj_axis = obj.x
    elif axis.upper() == "Y":
        obj_axis = obj.y
    elif axis.upper() == "Z":
        obj_axis = obj.z
    return obj_axis


def get_sorted_axis(axis, obj):
    if axis.upper() == "X":
        obj_interest_axis = obj.x
        obj_axis1 = obj.y
        obj_axis2 = obj.z
    elif axis.upper() == "Y":
        obj_interest_axis = obj.y
        obj_axis1 = obj.x
        obj_axis2 = obj.z
    elif axis.upper() == "Z":
        obj_interest_axis = obj.z
        obj_axis1 = obj.x
        obj_axis2 = obj.y
    return obj_interest_axis, obj_axis1, obj_axis2


def get_sorted_axis_name(axis):
    if axis.upper() == "X":
        obj_axis1 = "Y"
        obj_axis2 = "Z"
    elif axis.upper() == "Y":
        obj_axis1 = "X"
        obj_axis2 = "Z"
    elif axis.upper() == "Z":
        obj_axis1 = "X"
        obj_axis2 = "Y"
    return obj_axis1, obj_axis2


def check_mf(f, mf):
    if type(mf) == int:
        mf = np.array([mf])
    else:
        mf = np.array(mf)
        
    # if mf > f or mf < -f:
    if any(mf > f) or any(mf < -f):
        raise ValueError("m_F should be in the interval [-F,F]")
    else:
        return f, mf


def is_first(axe, plane):
    if axe == plane[0]:
        return True
    else:
        return False


def is_second(axe, plane):
    if axe == plane[1]:
        return True
    else:
        return False


def cyclic_perm(a):
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        if j != 0:
            file.write("\033[F")  # back to previous line
            file.write("\033[K")  # delete line
        file.write("%s[%s%s] %i %%" % (prefix, "#" * x, "." * (size - x), j / count * 100))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        if i % 4 == 0:
            show(i + 1)
    file.write("\n")
    file.flush()


def progressbar_enumerate(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        if j != 0:
            file.write("\033[F")  # back to previous line
            file.write("\033[K")  # delete line
        file.write("%s[%s%s] %i %%" % (prefix, "#" * x, "." * (size - x), j / count * 100))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield i, item  # added i
        if i % 4 == 0:
            show(i + 1)
    file.write("\n")
    file.flush()


def vec_to_string(vec):
    f = (len(vec) - 1) / 2
    mfs = np.arange(-f, f + 1, dtype=int)
    s = ""
    for idx, elt in enumerate(vec):
        mf = abs(mfs[idx])
        if mfs[idx] < 0:
            s += f"{elt:.2f}m$_{{- {mf} }}$+"
        else:
            s += f"{elt:.2f}m$_{mf}$+"
    return s[:-1]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def nan_to_zeros(*args):
    """Takes as many arrays as wanted and replace nans by zeros"""
    for arr in args:
        where_are_NaNs = np.isnan(arr)
        arr[where_are_NaNs] = 0


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def findKClosestElements(nums, k, target):
    left = 0
    right = len(nums) - 1
 
    while right - left >= k:
        if abs(nums[left] - target) > abs(nums[right] - target):
            left = left + 1
        else:
            right = right - 1
    return nums[left:left + k]