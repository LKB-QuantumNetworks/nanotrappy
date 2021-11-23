from nanotrappy.utils.utils import progressbar
import numpy as np
from numpy.core.defchararray import index
from copy import copy


class Geometry:

    dimension = None

    def __init__(self, name=""):
        self.name = name

    def get_dimension(self):
        return self.dimension

    def isAxis(self):
        return self.dimension == 1

    def isPlane(self):
        return self.dimension == 2


################################################################################################################
##################################################### Axes #####################################################
################################################################################################################


class Axis(Geometry):

    dimension = 1

    def __init__(self, name="", coordinates=(0, 0), sign=1):
        self.name = name
        self.coordinates = coordinates
        self._normal_plane = None
        self._index = None
        self._sign = sign

    @property
    def index(self):
        return self._index

    @property
    def sign(self):
        return self._sign

    @property
    def normal_plane(self):
        return self._normal_plane

    def set_coordinates(self, coord1, coord2):
        self.coordinates = (coord1, coord2)

    def coordinates_indexes(self, array1, array2):
        return np.argmin(np.abs(array1 - self.coordinates[0])), np.argmin(np.abs(array2 - self.coordinates[1]))

    def fetch_in(self, simulation):
        return getattr(simulation, self.name)

    def to_normal_vector(self):
        temp = [0, 0, 0]
        temp[self.index] = self.sign
        return temp

    def __neg__(self):
        ax = copy(self)
        ax._sign = -ax._sign
        return ax


class AxisX(Axis):
    def __init__(self, coordinates=(0, 0)):
        super().__init__("x", coordinates=coordinates)
        self._index = 0

    @property
    def normal_plane(self):
        return PlaneYZ()

    def to_slice(self, index_coord_1, index_coord_2):
        return np.s_[:, :, index_coord_1, index_coord_2]

    def complete_orthogonal_basis(self, position=0):
        return AxisY(coordinates=(position, self.coordinates[1])), AxisZ(coordinates=(position, self.coordinates[0]))


class AxisY(Axis):
    def __init__(self, coordinates=(0, 0)):
        super().__init__("y", coordinates=coordinates)
        self._index = 1

    @property
    def normal_plane(self):
        return PlaneXZ()

    def to_slice(self, index_coord_1, index_coord_2):
        return np.s_[:, index_coord_1, :, index_coord_2]

    def complete_orthogonal_basis(self, position=0):
        return AxisX(coordinates=(position, self.coordinates[1])), AxisZ(coordinates=(self.coordinates[0], position))


class AxisZ(Axis):
    def __init__(self, coordinates=(0, 0)):
        super().__init__("z", coordinates=coordinates)
        self._index = 2

    @property
    def normal_plane(self):
        return PlaneXY()

    def to_slice(self, index_coord_1, index_coord_2):
        return np.s_[:, index_coord_1, index_coord_2, :]

    def complete_orthogonal_basis(self, position=0):
        return AxisX(coordinates=(self.coordinates[1], position)), AxisY(coordinates=(self.coordinates[0], position))


################################################################################################################
#################################################### Planes ####################################################
################################################################################################################


class Plane(Geometry):

    dimension = 2

    def __init__(self, name, normal_axis, normal_coord=0):
        self.name = name
        self._normal_axis = normal_axis
        self.normal_coord = normal_coord

    @property
    def normal_axis(self):
        return self._normal_axis


class PlaneXY(Plane):
    def __init__(self, normal_coord=0):
        super().__init__(name="xy", normal_axis=AxisZ(), normal_coord=normal_coord)

    def get_base_axes(self):
        # return AxisX(), AxisY()
        return AxisX(coordinates=(0, self.normal_coord)), AxisY(coordinates=(0, self.normal_coord))


class PlaneYZ(Plane):
    def __init__(self, normal_coord=0):
        super().__init__(name="yz", normal_axis=AxisX(), normal_coord=normal_coord)

    def get_base_axes(self):
        # return AxisY(), AxisZ()
        return AxisY(coordinates=(self.normal_coord, 0)), AxisZ(coordinates=(self.normal_coord, 0))


class PlaneXZ(Plane):
    def __init__(self, normal_coord=0):
        super().__init__(name="xz", normal_axis=AxisY(), normal_coord=normal_coord)

    def get_base_axes(self):
        return AxisX(coordinates=(self.normal_coord, 0)), AxisZ(coordinates=(0, self.normal_coord))


################################################################################################################
################################################### Processor ##################################################
################################################################################################################


class DimensionNotAllowed(Exception):
    pass


class DimensionMismatch(Exception):
    pass


class FormattingError(Exception):
    pass


class FieldGeometryProcessor:
    @classmethod
    def restrict(self, simulation, geometry):
        E = [np.array([]) for k in range(len(simulation.E))]

        geometry_dimension = geometry.get_dimension()
        data_dimension = simulation.get_data_dimension()

        if data_dimension != 3:
            raise FormattingError(
                "You must give the electric fields in all 3 directions (one dimension can be of size 1)"
            )
        if data_dimension < geometry_dimension:
            raise DimensionMismatch("The data provided do not allow to compute the potential on the chosen geometry.")
        elif data_dimension == geometry_dimension:
            return simulation.E.copy()
        else:
            if geometry_dimension == 1:

                axis1, axis2 = geometry.normal_plane.get_base_axes()
                # coord1, coord2 = getattr(simulation, axis1.name), getattr(simulation, axis2.name)
                coord1, coord2 = axis1.fetch_in(simulation), axis2.fetch_in(simulation)
                index_coord_1, index_coord_2 = geometry.coordinates_indexes(coord1, coord2)

                slice = geometry.to_slice(index_coord_1, index_coord_2)
                # print(slice)

                for j in range(len(simulation.E)):  # number of wavelengths
                    E[j] = simulation.E[j][slice]

                return E

            elif geometry_dimension == 2:

                # index_coord = np.argmin(np.abs(getattr(simulation, geometry.normal_axis.name) - geometry.normal_coord))
                index_coord = np.argmin(np.abs(geometry.normal_axis.fetch_in(simulation) - geometry.normal_coord))
                for j in range(len(simulation.E)):  # number of wavelengths
                    E[j] = simulation.E[j].take(index_coord, axis=geometry.normal_axis.index + 1)

                return E

            else:
                raise DimensionNotAllowed(
                    "The dimension of the specified geometry is not taken in charge (yet) by nanotrappy"
                )


class SimulTest:
    def __init__(self, geometry=Axis()):
        self.x = np.arange(start=0, stop=100, step=1)
        self.y = np.arange(start=0, stop=101, step=1)
        self.z = np.arange(start=0, stop=102, step=1)
        self.E = np.random.rand(2, 3, 10, 11, 12)

        self.geometry = geometry

    def get_data_dimension(self):
        return np.ndim(self.E[0]) - 1


# if __name__ == "__main__":
#     ax1 = AxisY(coordinates=(2, 4))
#     pl1 = PlaneYZ()

#     print((-ax1).to_normal_vector())

#     s = SimulTest()

#     print(ax1.fetch_in(s))

#     el = FieldGeometryProcessor.restrict(s, pl1)

#     print(el[0].shape[1:])

#     size = el[0].shape[1:]
#     aa = np.random.rand(*(5, *size, 7))
#     print(aa.shape)

#     import itertools

#     # ranges = [progressbar(range(0, 5), "\n Computing: ", 40), range(3, 7)]
#     # for xs in itertools.product(*ranges):
#     #     print(aa[(0, *xs)])

#     ranges = [range(s) for s in size]
#     ranges[0] = progressbar(ranges[0], "\n Computing: ", 40)
#     rt = itertools.product(*ranges)
#     print(rt)
#     import time

#     # for idx in itertools.product(*ranges):
#     #     time.sleep(2)

if __name__ == "__main__":
    ax = AxisX(coordinates=(0, 0))
    ax1, ax2 = ax.complete_orthogonal_basis(position=10)
    print(ax1.name, ax1.coordinates)
    print(ax2.name, ax2.coordinates)
