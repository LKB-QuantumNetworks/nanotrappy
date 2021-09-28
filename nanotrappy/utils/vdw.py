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


class PlaneSurface(Surface):
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


class CylindricalSurface(Surface):
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


class NoSurface(Surface):
    def __init__(self):
        self.params = {"type": type(self).__name__}
        pass

    def distance(self, point):
        return 0


if __name__ == "__main__":
    pl = PlaneSurface(normal_axis=AxisX(), normal_coord=0)

    print(pl.distance((11.2, 33.6, 4)))
