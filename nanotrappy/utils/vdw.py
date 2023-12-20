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
        dot_product = np.dot(self.normal_axis.to_normal_vector(), axis.to_normal_vector())
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
    
class SlabSurface(Surface):
    __desc__ = "slab"
    
    def __init__(self, normal_axis, finite_axis, width):
        #Give normal_coord at the center of the slab
        self.finite_axis = finite_axis
        self.normal_axis = normal_axis
        # self.normal_coords = normal_coords
        self.width = width
        self.params = {
            "type": type(self).__name__,
            "normal_axis": str(self.normal_axis.name),
            "finite_axis": str(self.finite_axis.name),
            # "normal_coord": str(self.normal_coord),
            "width":str(self.width)
        }
        
    def U_CP(self, y,z, tz = 75e-9):
        Y,Z = np.meshgrid(y,z)
        eps = 1e-11 #small correction in case there is a z = width/2
        U = - np.heaviside(y,0)* 6/8*((2 - (np.sqrt(1/(Y**2+(tz-Z)**2))*(2*Y**4+Y**2*(tz-Z)**2+2*(tz-Z)**4))/Y**3)/(3*(tz-Z+eps)**3) + (2 - (np.sqrt(1/(Y**2+(tz+Z)**2))*(2*Y**4+Y**2*(tz+Z)**2+2*(tz+Z)**4))/Y**3)/(3*(tz+Z+eps)**3))
        return np.squeeze(U)
    
    def set_mask(self, simul):
        dimension = simul.geometry.get_dimension()
        if dimension == 2:
            axis1, axis2 = simul.geometry.get_base_axes()
            x, y = axis1.fetch_in(simul), axis2.fetch_in(simul)
            
            if axis1.name == self.finite_axis.name and axis2.name == self.normal_axis.name:
                mask = np.transpose(np.array(self.U_CP(y, x, self.width/2)))
                
            if axis2.name == self.finite_axis.name and axis1.name == self.normal_axis.name:
                mask = np.transpose(np.array(self.U_CP(x, y, self.width/2)))
                
            if axis1.name == self.finite_axis.name and axis2.name != self.normal_axis.name:
                mask = np.array(self.U_CP(simul.geometry.normal_coord, x, self.width/2))
                mask, _ = np.meshgrid(y, mask)
                
            if axis2.name == self.finite_axis.name and axis1.name != self.normal_axis.name:
                mask = np.array(self.U_CP(simul.geometry.normal_coord, y, self.width/2))
                mask, _ = np.meshgrid(mask, x)
                
            if axis2.name != self.finite_axis.name and axis1.name == self.normal_axis.name:
                mask = np.array(self.U_CP(x, simul.geometry.normal_coord, self.width/2))
                mask, _ = np.meshgrid(y, mask)

            if axis1.name != self.finite_axis.name and axis2.name == self.normal_axis.name:
                mask = np.array(self.U_CP(y, simul.geometry.normal_coord, self.width/2))
                mask, _ = np.meshgrid(mask, x)            
            return mask
        
class SlabSurface_noH(Surface):
    __desc__ = "slab"
    
    def __init__(self, normal_axis, finite_axis, width, shift = 0):
        #Give normal_coord at the center of the slab
        self.finite_axis = finite_axis
        self.normal_axis = normal_axis
        # self.normal_coords = normal_coords
        self.width = width
        self.params = {
            "type": type(self).__name__,
            "normal_axis": str(self.normal_axis.name),
            "finite_axis": str(self.finite_axis.name),
            # "normal_coord": str(self.normal_coord),
            "width":str(self.width)
        }
        
    # def U_CP(self, y,z, tz = 75e-9):
    #     Y,Z = np.meshgrid(y,z)
    #     eps = 1e-11 #small correction in case there is a z = width/2
    #     U = - 6/8*((2 - (np.sqrt(1/(Y**2+(tz-Z)**2))*(2*Y**4+Y**2*(tz-Z)**2+2*(tz-Z)**4))/Y**3)/(3*(tz-Z+eps)**3) + (2 - (np.sqrt(1/(Y**2+(tz+Z)**2))*(2*Y**4+Y**2*(tz+Z)**2+2*(tz+Z)**4))/Y**3)/(3*(tz+Z+eps)**3))
    #     return np.squeeze(U)
    
    def U_CP(self, y, z, tz = 75e-9):
        d, a = np.meshgrid(y,z+1e-10)
        d, a = np.squeeze(d), np.squeeze(a)
        U_math = 6/np.pi*(np.pi*(2*a**7*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) - np.sqrt(
        a**2 + d**2 + 2*a*tz + tz**2)) + a**5*(d**2 - 6*tz**2)*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) 
        - np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) + a*tz**2*(6*d**4 + d**2*tz**2 - 2*tz**4)*
        (np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) - np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) + 
         2*a**3*(d**4 - d**2*tz**2 + 3*tz**4)*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) 
        - np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) + 2*a**6*tz*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) 
       + np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) - a**4*tz*(d**2 + 6*tz**2)*
       (np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) + np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) - 
         tz**3*(-4*d**3*np.sqrt(a**4 + 2*a**2*(d**2 - tz**2) + (d**2 + tz**2)**2) + 
        2*d**4*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) + np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) + 
        d**2*tz**2*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) + np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) + 
        2*tz**4*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) + np.sqrt(a**2 + d**2 + 2*a*tz + tz**2))) + 
         2*a**2*tz*(6*d**3*np.sqrt(a**4 + 2*a**2*(d**2 - tz**2) + (d**2 + tz**2)**2) - 
        3*d**4*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) + np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) + 
        d**2*tz**2*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) + np.sqrt(a**2 + d**2 + 2*a*tz + tz**2)) + 
        3*tz**4*(np.sqrt(a**2 + d**2 - 2*a*tz + tz**2) + np.sqrt(a**2 + d**2 + 
        2*a*tz + tz**2)))))/(24*d**3*(a - tz)**3*(a + tz)**3*np.sqrt(a**4 + 2*a**2*(d**2 - tz**2) + (d**2 + tz**2)**2))
        return U_math
    
    def set_mask(self, simul):
        dimension = simul.geometry.get_dimension()
        if dimension == 2:
            axis1, axis2 = simul.geometry.get_base_axes()
            x, y = axis1.fetch_in(simul), axis2.fetch_in(simul)
            
            if axis1.name == self.finite_axis.name and axis2.name == self.normal_axis.name:
                mask = np.transpose(np.array(self.U_CP(y, x, self.width/2)))
                
            if axis2.name == self.finite_axis.name and axis1.name == self.normal_axis.name:
                mask = np.transpose(np.array(self.U_CP(x, y, self.width/2)))
                
            if axis1.name == self.finite_axis.name and axis2.name != self.normal_axis.name:
                mask = np.array(self.U_CP(simul.geometry.normal_coord, x, self.width/2))
                mask, _ = np.meshgrid(y, mask)
                
            if axis2.name == self.finite_axis.name and axis1.name != self.normal_axis.name:
                mask = np.array(self.U_CP(simul.geometry.normal_coord, y, self.width/2))
                mask, _ = np.meshgrid(mask, x)
                
            if axis2.name != self.finite_axis.name and axis1.name == self.normal_axis.name:
                mask = np.array(self.U_CP(x, simul.geometry.normal_coord, self.width/2))
                mask, _ = np.meshgrid(y, mask)

            if axis1.name != self.finite_axis.name and axis2.name == self.normal_axis.name:
                mask = np.array(self.U_CP(y, simul.geometry.normal_coord, self.width/2))
                mask, _ = np.meshgrid(mask, x)            
            return mask
    

class NoSurface(Surface):
    __desc__ = "nosurface"

    def __init__(self):
        self.params = {"type": type(self).__name__}
        pass

    def distance(self, point):
        return 0

    def get_slab(self, axis_data, trap_data, simul, axis, manual_edge=None):
        if manual_edge is None:
            manual_edge = min(axis_data)
            # raise ValueError(
            #     "No surface for CP interactions have been specified. To restrict the search for the minimum in the right zone, you have to specify an edge"
            # )
        edge = manual_edge
        index_edge = np.argmin(np.abs(axis_data - edge))
        
        y_outside = axis_data[index_edge:] - edge
        trap_outside = trap_data[index_edge:]
        return y_outside, trap_outside


if __name__ == "__main__":
    pl = PlaneSurface(normal_axis=AxisX(), normal_coord=0)

    print(pl.distance((11.2, 33.6, 4)))
    print(pl.isCylindricalSurface())
