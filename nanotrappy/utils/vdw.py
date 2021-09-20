import numpy as np
from nanotrappy.utils.utils import *

class Surface:
    def __init__(self):
        pass
    
    def set_mask2D(self, simul, plane, coord):
        x,y = set_axis_from_plane(plane,simul)
        mask = np.zeros((len(x),len(y)))
        coord_x, coord_y, coord_z = set_axis_index_from_plane(plane)
        point = np.zeros((3,))
        point[coord_z] = coord
        for k in range(len(x)):
            for j in range(len(y)):
                point[coord_x] = x[k]
                point[coord_y] = y[j]
                d = self.distance(point)
                if d<=0:
                    mask[k,j] = 0
                else:
                    mask[k,j] = 1/d**3
        return mask
    
    def set_mask1D(self, simul, axis, coord1, coord2):
        x = set_axis_from_axis(axis, simul)
        mask = np.zeros((len(x)))
        coord_x, coord_y, coord_z = set_axis_index_from_axis(axis)
        point = np.zeros((3,))
        point[coord_y] = coord1
        point[coord_z] = coord2
        for k in range(len(x)):
            point[coord_x] = x[k]
            d = self.distance(point)
            if d<=0:
                mask[k] = 0
            else:
                mask[k] = 1/d**3
        return mask
            
        
class Plane(Surface):
    def __init__(self, vect, coord):
        self.vect_n = vect
        self.coord = coord
        
        self.a = self.vect_n[0]
        self.b = self.vect_n[1]
        self.c = self.vect_n[2] 
        self.d = - self.a*coord[0] - self.b*coord[1] - self.c*coord[2]
        self.params = {"type" : type(self).__name__, "normal vector" : str(self.vect_n), "coord" : str(self.coord)}
        

    def distance(self, point): 
        d = (self.a*point[0] + self.b*point[1] + self.c*point[2] + self.d)/ (np.sqrt(self.a**2 + self.b**2  + self.c**2))
        return d
    
class Cylinder(Surface):
    def __init__(self, center, radius, propagation_axis):
        self.center = center
        self.radius = radius
        self.propagation_axis = set_axis_index(propagation_axis)
        self.params = {"type" : type(self).__name__, "center" : str(self.center), "radius" : str(self.radius), "propagation axis" : self.propagation_axis}

        
    def distance(self, point): 
        indices = [0,1,2]
        indices.pop(self.propagation_axis)
        r = np.sqrt((point[indices[0]]-self.center[indices[0]])**2 + (point[indices[1]]-self.center[indices[1]])**2)
        d = r - self.radius
        return d
       
class NoSurface(Surface):
    def __init__(self):
        self.params = {"type" : type(self).__name__}
        pass
    
    def distance(self,point):
        return 0
