from nanotrappy.trapping.geometry import AxisX
import numpy as np
from nanotrappy.utils.utils import *


class Surface:
    def __init__(self):
        pass

    def set_mask(self, simul):
        dimension = simul.geometry.get_dimension()

        if dimension == 1:
            axis = simul.geometry
            x = axis.fetch_in(simul)
            mask = np.zeros((len(x)))

            axis1, axis2 = simul.geometry.normal_plane.get_base_axes()
            idx1, idx2 = axis1.index, axis2.index

            point = np.zeros((3,))
            point[idx1] = simul.geometry.coordinates[0]
            point[idx2] = simul.geometry.coordinates[1]

            for k, _ in enumerate(x):
                point[simul.geometry.index] = x[k]
                d = self.distance(point)
                if d <= 0:
                    mask[k] = 0
                else:
                    mask[k] = 1 / d ** 3
            return mask

        elif dimension == 2:
            axis1, axis2 = simul.geometry.get_base_axes()
            x, y = axis1.fetch_in(simul), axis2.fetch_in(simul)

            mask = np.zeros((len(x), len(y)))

            point = np.zeros((3,))
            point[simul.geometry.normal_axis.index] = simul.geometry.normal_coord
            for k, _ in enumerate(x):
                for j, _ in enumerate(y):
                    point[axis1.index] = x[k]
                    point[axis2.index] = y[j]
                    d = self.distance(point)
                    if d <= 0:
                        mask[k, j] = 0
                    else:
                        mask[k, j] = 1 / d ** 3
            return mask

        else:
            pass

    def isPlaneSurface(self):
        return True if self.__desc__ == "planesurface" else False

    def isCylindricalSurface(self):
        return True if self.__desc__ == "cylindricalsurface" else False

    def isNoSurface(self):
        return True if self.__desc__ == "nosurface" else False


class PlaneSurface(Surface):
    __desc__ = "planesurface"

    def __init__(self, normal_axis, normal_coord):
        self.normal_axis = normal_axis
        self.normal_coord = normal_coord
        self.params = {
            "type": type(self).__name__,
            "normal_axis": str(self.normal_axis.name),
            "normal_coord": str(self.normal_coord),
        }

    def distance(self, point):
        # normal_vect = [0, 0, 0]
        # normal_vect[self.normal_axis.index] = 1
        normal_vect = self.normal_axis.to_normal_vector()

        point_in_plane = [0, 0, 0]
        point_in_plane[self.normal_axis.index] = self.normal_coord

        D = (
            -normal_vect[0] * point_in_plane[0]
            - normal_vect[1] * point_in_plane[1]
            - normal_vect[2] * point_in_plane[2]
        )

        d = (normal_vect[0] * point[0] + normal_vect[1] * point[1] + normal_vect[2] * point[2] + D) / (
            np.sqrt(normal_vect[0] ** 2 + normal_vect[1] ** 2 + normal_vect[2] ** 2)
        )
        return d

    def get_slab(self, axis_data, trap_data, simul, axis, manual_edge=None):
        dot_product = np.dot(self.normal_axis.to_normal_vector(), axis)
        if abs(dot_product) != 1:
            return -1, trap_data
        else:
            axis_data = axis.fetch_in(simul)
            idx = find_nearest(axis_data, self.normal_coord)
            if np.abs(axis_data[idx] - self.normal_coord) > np.max(np.diff(axis_data)):  ## very ugly, but kept for now
                return -1, trap_data
            if dot_product == 1:
                return axis_data[idx:], trap_data[idx:]
            else:
                return axis_data[: idx + 1], trap_data[: idx + 1]


class CylindricalSurface(Surface):

    __desc__ = "cylindricalsurface"

    def __init__(self, axis, radius):
        self.axis = axis
        self.radius = radius
        self.params = {
            "type": type(self).__name__,
            "center": str(self.axis.coordinates),
            "radius": str(self.radius),
            "propagation axis": self.axis.name,
        }

    def distance(self, point):
        axis1, axis2 = self.axis.normal_plane.get_base_axes()
        idx1, idx2 = axis1.index, axis2.index
        coord_center_1, coord_center_2 = self.axis.coordinates
        r = np.sqrt((point[idx1] - coord_center_1) ** 2 + (point[idx2] - coord_center_2) ** 2)
        d = r - self.radius
        return d

    def get_slab(self, axis_data, trap_data, simul, axis, manual_edge=None):
        ax1, ax2 = self.axis.normal_plane.get_base_axes()
        if np.abs(np.dot(ax1.to_normal_vector(), axis.to_normal_vector())) == 1:
            ax = ax1
            center_coord = self.axis.coordinates[0]
        elif np.abs(np.dot(ax2.to_normal_vector(), axis.to_normal_vector())) == 1:
            ax = ax2
            center_coord = self.axis.coordinates[1]
        else:
            raise ValueError("Fatal Error")

        idx = find_nearest(axis_data, center_coord + self.radius)
        return axis_data[idx + 1 :], trap_data[idx + 1 :]


class NoSurface(Surface):
    __desc__ = "nosurface"

    def __init__(self):
        self.params = {"type": type(self).__name__}
        pass

    def distance(self, point):
        return 0

    def get_slab(self, trap_data, simul, axis_data, manual_edge=None):
        if manual_edge is None:
            raise ValueError(
                "No surface for CP interactions have been specified. To restrict the search for the minimum in the right zone, you have to specify an edge"
            )
        edge = manual_edge
        index_edge = np.argmin(np.abs(axis_data - edge))
        y_outside = axis_data[index_edge:] - edge
        trap_outside = trap_data[index_edge:]


if __name__ == "__main__":
    pl = PlaneSurface(normal_axis=AxisX(), normal_coord=0)

    print(pl.distance((11.2, 33.6, 4)))
    print(pl.isCylindricalSurface())
