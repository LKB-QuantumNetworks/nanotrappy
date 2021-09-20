import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from operator import itemgetter
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.signal import find_peaks
from nanotrappy.utils.physicalunits import *
from nanotrappy.utils.utils import *



class Viz():
    def __init__(self, simul,trapping_axis):
        #Trapping_axis is the one perpendicular to the surface if one is defined
        self.trapping_axis = trapping_axis
        self.simul = simul
        
    def plot_trap(self,plane,mf = 0,Pranges=[10,10]):
        """Shows a 2D plot of the total potential with power sliders
        Only available if a 2D simulation has been run.
    
        Args:
            simul (Simulation object): A simulation object with computation of potentials already run.
            plane (str): As we are dealing with 2D plots, we have to specify the plane we are looking at to choose the right coordinates for plotting.
            mf (int): Mixed mf state we want to plot. In 2D we can only specify one integer. Default to 0.
            Pranges (list): List with the maximum values of the beam powers we want to display on the sliders. Defaults to [10,10]     
        Raise:
            TypeError: if only a 1D computation of the potential has been done before plotting.
            
        Returns:
            (tuple): containing:
                
                - fig: figure
                - ax: axis of figure
                - slider_ax: sliders (needed for interactivity of the sliders)
    
        """    
        if np.ndim(simul.total_potential()) <= 2:
            raise TypeError("This method can only be used if a 2D computation of the potential has been done")
        
        if len(Pranges) != len(simul.trap.beams):
            raise ValueError("When specifying the upper ranges of P for plotting, you have to give as many as many values as there are beams.")
        
        _, mf = check_mf(simul.atomicsystem.f,mf)
        coord1, coord2 = set_axis_from_plane(plane, simul)
        mf_index = int(mf + simul.atomicsystem.f)
        trap = np.real(simul.total_potential())[:,:,mf_index]
        trap_noCP = np.real(simul.total_potential_noCP[:,:,mf_index])
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.5, bottom=0.05)
        #the norm TwoSlopeNorm allows to fix the 0 of potential to the white color, so that we can easily distinguish between positive and negative values of the potential
        a = plt.pcolormesh(coord1,coord2,np.transpose(trap),shading = "gouraud",norm = colors.TwoSlopeNorm(vmin=min(np.min(trap_noCP)/2,-0.001), vcenter=0, vmax=max(np.max(trap_noCP)*2,0.001)), cmap = "seismic_r")
        cbar = plt.colorbar(a)
        cbar.set_label("Total potential (mW)", rotation = 270, labelpad = 6)
        # y_edge_ind = [i for i in range(len(simul.y)) if simul.y[i] >= edge]
        # index_min = np.unravel_index(trap[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
        # point, = plt.plot(simul.x[index_min[0]],simul.y[index_min[1]+y_edge_ind[0]],'ro')
        plt.xlabel("%s (nm)" %(plane[0]))
        plt.ylabel("%s (nm)" %(plane[1]))
    
        plt.title("2D plot of trapping potential in the %s plane" %(plane))
    
        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'   
        slider_ax = []
        axes = []
        
        for (k,beam) in enumerate(simul.trap.beams):
            axes.append(plt.axes([0.15 + k*0.08, 0.1, 0.03, 0.75], facecolor=axcolor))
            slider_ax.append(Slider(axes[k], 'Power \n Beam %s' %(k+1), 0, Pranges[k], valinit=np.mean(beam.get_power()), valstep=0.1,orientation='vertical'))
            # axPred = plt.axes([0.23, 0.1, 0.03, 0.75], facecolor=axcolor)
            # sPred = Slider(axPred, 'Power \n Beam 2', 0, 50.0, valinit=np.mean(simul.trap.beams[1].get_power()), valstep=0.1,orientation='vertical')
    
    
    #     axlmbdablue = plt.axes([0.31, 0.1, 0.03, 0.75], facecolor=axcolor)
    #     slmbdablue = DiscreteSlider(axlmbdablue, 'lmbdablue', min(simul.lmbdas_modes)/nm, max(simul.lmbdas_modes)/nm, allowed_vals= simul.lmbdas_modes/nm, valinit=simul.lmbdas_modes[0],valfmt='%1.3f',orientation='vertical')
    #     axlmbdared = plt.axes([0.39, 0.1, 0.03, 0.75], facecolor=axcolor)
    #     slmbdared = DiscreteSlider(axlmbdared, 'lmbdared', min(simul.lmbdas_modes)/nm, max(simul.lmbdas_modes)/nm, allowed_vals= simul.lmbdas_modes/nm, valinit=simul.lmbdas_modes[-1],valfmt='%1.3f',orientation='vertical')
    
        def updateP(val):
            P = []
            for (k,slider) in enumerate(slider_ax):
                P.append(slider.val*mW)
            simul.trap.set_powers(P)
            trap_2D = simul.total_potential()[:,:,mf_index]
            a.set_array(np.transpose(np.real(simul.total_potential_noCP[:,:,mf_index])).ravel())
            a.autoscale()
            a.set_array(np.transpose(np.real(trap_2D)).ravel())
            # if plane == "XY":
            #     argmin_update = np.unravel_index(trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
            #     point.set_data(simul.x[argmin_update[0]],simul.y[argmin_update[1]+y_edge_ind[0]])
    
            # elif plane == "YZ":
            #     argmin_update = np.unravel_index(trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
            #     point.set_data(simul.y[argmin_update[0]+y_edge_ind[0]],simul.z[argmin_update[1]])
            # print(argmin_update)
            # a.autoscale()
            fig.canvas.draw_idle()
            
    #     def updatel(val):
    #         print("lambdablue", slmbdablue.val)
    #         if simul.trap.nblue != 0:
    #             simul.wavelength_index_blue = np.argmin(np.abs(np.array(simul.wavelengths_dict[0])/nm - slmbdablue.val))
    #         if simul.trap.nred != 0:
    #             simul.wavelength_index_red = np.argmin(np.abs(np.array(simul.wavelengths_dict[1])/nm - slmbdared.val))
    #         print("wb, wr", simul.wavelength_index_blue, simul.wavelength_index_red)
    #         trap_2D = simul.simulate(simul.atomicsystem.groundstate,simul.f,simul.mf,plane,simul.out_of_plane_coord,simul.wavelength_index_blue,simul.wavelength_index_red)
    #         a.set_array(np.transpose(np.real(trap_2D)).ravel())
    # #            a.set_clim(vmin=np.min(np.real(trap_2D)),vmax=np.max(np.real(trap_2D)))
    # #            cbar.draw_all()
    # #            a.set_data(np.transpose(np.real(simul.trap_2D[plane])))
    #         if plane == "XY":
    #             argmin_update = np.unravel_index(trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].argmin(),trap_2D[:,y_edge_ind[0]:y_edge_ind[-1]+1].shape)
    #             print(argmin_update)
    #             point.set_data(simul.x[argmin_update[0]],simul.y[argmin_update[1]+y_edge_ind[0]])
    
    #         elif plane == "YZ":
    #             argmin_update = np.unravel_index(trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].argmin(),trap_2D[y_edge_ind[0]:y_edge_ind[-1]+1,:].shape)
    #             point.set_data(simul.y[argmin_update[0]+y_edge_ind[0]],simul.z[argmin_update[1]])
    #             print(argmin_update)
    #         # print(argmin_update)
    #         a.autoscale()
    #         fig.canvas.draw_idle()
        for slider in slider_ax :
            slider.on_changed(updateP)
    #     slmbdared.on_changed(updatel)
    #     slmbdablue.on_changed(updatel)
        plt.show()
        
        return fig, ax, slider_ax#, slmbdablue, slmbdared
        
    def plot_trap1D(simul,axis,mf = 0, Pranges=[10,10]):
        """Shows a 1D plot of the total potential with power sliders
        Only available if a 1D simulation has been run.
    
        Args:
            simul (Simulation object): A simulation object with computation of 1D potentials already run.
            plane (str): As we are dealing with 1D plots, we have to specify the axis along which we are looking the trap.
            mf (int or list): Mixed mf state we want to plot. If a list is given, all the specified mf states will be showed. Default to 0.
            Pranges (list): List with the maximum values of the beam powers we want to display on the sliders. Defaults to [10,10] 
        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.
            
        Returns:
            (tuple): containing:
                
                - fig: figure
                - ax: axis of figure
                - slider_ax: sliders (needed for interactivity of the sliders)
        """
        
        if np.ndim(simul.total_potential()) >= 3:
            raise TypeError("This method can only be used if a 1D computation of the potential has been done.")
        if len(Pranges) != len(simul.trap.beams):
            raise ValueError("When specifying the upper ranges of P for plotting, you have to give as many as many values as there are beams.")
        
        _, mf = check_mf(simul.atomicsystem.f,mf)
        
        x = set_axis_from_axis(axis, simul)
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.27)
        jet = cm = plt.get_cmap('Greys')
        cNorm  = colors.Normalize(vmin=-1, vmax=len(mf))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        a = []
        
        mf_index = mf + [simul.atomicsystem.f]
       
        trap = np.real(simul.total_potential())
        trap_noCP = np.real(simul.total_potential_noCP)
        
        for k in range(len(mf_index)):
            colorVal = scalarMap.to_rgba(k)
            a = a + plt.plot(x,trap[:,mf_index[k]], color = colorVal, label = "m$_f$ = %s" %(mf[k]), linewidth = 2 + 3/len(simul.mf_all))
    
        if len(mf) == 1 and len(simul.trap.beams) == 2:
            b, = plt.plot(x,np.real(simul.trap.beams[0].get_power()*np.real(simul.potentials[0,:,mf_index[0]])), color = 'blue', linewidth = 2)
            r, = plt.plot(x,np.real(simul.trap.beams[1].get_power()*np.real(simul.potentials[1,:,mf_index[0]])), color = 'red', linewidth = 2)
    
        plt.axhline(y=0, color='black', linestyle='--')
        # plt.ylim(-3,3)
        plt.legend()
        plt.xlabel(axis+" (nm)")
        plt.ylabel("E (mK)")
    #        plt.ylim(-5e-14,5e-14)
        plt.title("1D plot of trapping potential along %s " %(axis))
    
        axcolor = 'lightgoldenrodyellow'
        slider_ax = []
        axes = []
        for (k,beam) in enumerate(simul.trap.beams):
            axes.append(plt.axes([0.2, 0.15-k*0.1, 0.65, 0.03], facecolor=axcolor))
            slider_ax.append(Slider(axes[k], 'Power \n Beam %s' %(k+1), 0, Pranges[k], valinit=np.mean(beam.get_power()), valstep=0.01))
    
        # if len(Pranges) == 2:
        #     sPblue = Slider(axPblue, 'Power \n Beam 1 (mW)', Pranges[0], Pranges[1], valinit=simul.trap.beams[0].get_power(), valstep=0.001)
        #     sPred = Slider(axPred, 'Power \n Beam 2 (mW)', Pranges[0], Pranges[1], valinit=simul.trap.beams[1].get_power(), valstep=0.0001)
        # elif len(Pranges) == 4:
        #     sPblue = Slider(axPblue, 'Power \n Beam 1 (mW)', Pranges[0], Pranges[1], valinit=simul.trap.beams[0].get_power(), valstep=0.001)
        #     sPred = Slider(axPred, 'Power \n Beam 2 (mW)', Pranges[2], Pranges[3], valinit=simul.trap.beams[1].get_power(), valstep=0.0001)
        # else :
        #     print("Default ranges for powers")
        #     sPblue = Slider(axPblue, 'Power \n Beam 1 (mW)', 0, 0.5, valinit=simul.trap.beams[0].get_power(), valstep=0.001)
        #     sPred = Slider(axPred, 'Power \n Beam 2 (mW)', 0, 0.1, valinit=simul.trap.beams[1].get_power(), valstep=0.0001)
    
        def updateP(val):
            P = []
            for (k,slider) in enumerate(slider_ax):
                P.append(slider.val*mW)
            simul.trap.set_powers(P)
            trap = np.real(simul.total_potential())
            for k in range(len(mf)):
                trap_k = trap[:,mf_index[k]]
                a[k].set_ydata(trap_k)
    
            if len(mf) == 1 and len(simul.trap.beams) == 2:
                b.set_ydata(np.real(simul.trap.beams[0].get_power()*np.real(simul.potentials[0,:,mf_index[0]])))
                r.set_ydata(np.real(simul.trap.beams[1].get_power()*np.real(simul.potentials[1,:,mf_index[0]])))
            fig.canvas.draw_idle()
            
        for slider in slider_ax :
            slider.on_changed(updateP)
        plt.show()
        
        return fig, ax, slider_ax
    
    
    def plot_3axis(simul,axis_name,coord1,coord2,mf=0,Pranges=[10,10]):
        _, mf = check_mf(simul.atomicsystem.f,mf)
        mf_shift = mf + simul.atomicsystem.f
        axis_of_interest, axis1, axis2 = get_sorted_axis(axis_name, simul)
        axis1_name, axis2_name = get_sorted_axis_name(axis_name)
        axis_index, axis1_index, axis2_index = set_axis_index_from_axis(axis_name)
        
        if len(simul.E[0].shape) != 4 :
            print("3D Electric fields must be fed in the Simulation class in order to use this function")
        else : 
            trap_1D_Y_allw = np.squeeze(np.real(simul.compute_potential_1D(axis_name,coord1,coord2)))[:,mf_shift]
            ymin_ind, y_min, trap_depth, trap_prominence = get_min_trap(simul,axis_name,mf_shift)
            mf_index, edge, y_outside, trap_Y_outside = get_coord_trap_outside_structure(simul, axis_name)
            min_pos = np.zeros(3)
            min_pos[axis_index] = y_min + edge
            min_pos[axis1_index] = coord1
            min_pos[axis2_index] = coord2
            print(min_pos)
            print("y_min = ", y_min)
            omegax, omegay, omegaz = 0, 0, 0
            if y_min is not None:
                print(np.delete(min_pos,axis1_index))
                print(np.delete(min_pos,axis2_index))
                omegay = get_trapfreq(simul,axis_name)
                trap_1D_X_allw = np.squeeze(np.real(simul.compute_potential_1D(axis1_name,np.delete(min_pos,axis1_index)[0],np.delete(min_pos,axis1_index)[1])))[:,mf_shift]
                omegax = get_trapfreq(simul,axis1_name)
                trap_1D_Z_allw = np.squeeze(np.real(simul.compute_potential_1D(axis2_name,np.delete(min_pos,axis2_index)[0],np.delete(min_pos,axis2_index)[1])))[:,mf_shift]
                omegaz = get_trapfreq(simul,axis2_name)
    
            fig, ax  = plt.subplots(3, figsize = (15,10))
            plt.subplots_adjust(left=0.25)
            axcolor = 'lightgoldenrodyellow'
            props = dict(boxstyle='round', facecolor=axcolor, alpha=0.5)
            
            textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, ), r'$\omega_x=%.2f (kHz) $' % (omegax, ), r'$\omega_y=%.2f (kHz) $' % (omegay, ), r'$\omega_z=%.2f (kHz) $' % (omegaz, )))
            box = plt.text(- 0.4, 0.6, textstr, transform=ax[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
            
            #Define axes that will eventually receive the sliders
            # axPblue = fig.add_axes([0.05, 0.35, 0.02, 0.55], facecolor = axcolor)
            # axPred = fig.add_axes([0.1, 0.35, 0.02, 0.55], facecolor = axcolor)
            
            slider_ax = []
            axes = []
            for (k,beam) in enumerate(simul.trap.beams):
                axes.append(plt.axes([0.05+k*0.05,0.35, 0.02, 0.55], facecolor=axcolor))
                slider_ax.append(Slider(axes[k], 'Power \n Beam %s' %(k+1), 0, Pranges[k], valinit=np.mean(beam.get_power()), valstep=0.01, orientation = 'vertical'))
    
            # axlmbdablue = fig.add_axes([0.15, 0.35, 0.02, 0.55], facecolor = axcolor)
            # axlmbdared = fig.add_axes([0.2, 0.35, 0.02, 0.55], facecolor = axcolor)
    
            # if len(Pranges) == 2:
            #     sPblue = Slider(axPblue, 'Power \n Beam 1 (mW)', Pranges[0], Pranges[1], valinit=simul.trap.beams[0].get_power(), valstep=0.001,orientation='vertical')
            #     sPred = Slider(axPred, 'Power \n Beam 2 (mW)', Pranges[0], Pranges[1], valinit=simul.trap.beams[1].get_power(), valstep=0.0001,orientation='vertical')
            # elif len(Pranges) == 4:
            #     sPblue = Slider(axPblue, 'Power \n Beam 1 (mW)', Pranges[0], Pranges[1], valinit=simul.trap.beams[0].get_power(), valstep=0.001,orientation='vertical')
            #     sPred = Slider(axPred, 'Power \n Beam 2 (mW)', Pranges[2], Pranges[3], va+linit=simul.trap.beams[1].get_power(), valstep=0.0001,orientation='vertical')
            # else :
            #     print("Default ranges for powers")
            #     sPblue = Slider(axPblue, 'Power \n Beam 1 (mW)', 0, 0.5, valinit=simul.trap.beams[0].get_power(), valstep=0.001,orientation='vertical')
            #     sPred = Slider(axPred, 'Power \n Beam 2 (mW)', 0, 0.1, valinit=simul.trap.beams[1].get_power(), valstep=0.0001,orientation='vertical')
                
            # slmbdablue = DiscreteSlider(axlmbdablue, 'lmbdablue', valmin = min(self.wavelengths_dict[0])/pu.nm, valmax = max(self.wavelengths_dict[0])/pu.nm, allowed_vals= np.array(self.wavelengths_dict[0])/pu.nm, valinit=self.wavelengths_dict[0][self.wavelength_index_blue],valfmt='%1.3f', orientation='vertical')
            # slmbdared = DiscreteSlider(axlmbdared, 'lmbdared', min(self.wavelengths_dict[1])/pu.nm, max(self.wavelengths_dict[1])/pu.nm, allowed_vals= np.array(self.wavelengths_dict[1])/pu.nm, valinit=self.wavelengths_dict[1][self.wavelength_index_red],valfmt='%1.3f',orientation='vertical')
            
            ly, = ax[0].plot(y_outside,trap_Y_outside)
            ax[0].set_ylim([-2, 2])
            if y_min is not None:
                lx, = ax[1].plot(axis1,trap_1D_X_allw)
                lz, = ax[2].plot(axis2,trap_1D_Z_allw)
            else:
                lx, = ax[1].plot(axis1,np.zeros((len(axis1),)))
                lz, = ax[2].plot(axis2,np.zeros((len(axis1),)))
            ax[0].set_xlabel("y (m)")
            ax[1].set_xlabel("x (m)")
            ax[2].set_xlabel("z (m)")
            plt.axhline(y=0, color='black', linestyle='--')
            plt.title("Total dipole trap for mf = %s in the 3 directions" %(mf))
            
            def updateP(val):
                P = []
                for (k,slider) in enumerate(slider_ax):
                    P.append(slider.val*mW)
                simul.trap.set_powers(P)
                trap_1D_Y = np.squeeze(np.real(simul.compute_potential_1D(axis_name,coord1,coord2)))[:,mf_shift]
                ymin_ind, y_min, trap_depth, trap_prominence = get_min_trap(simul, axis_name)
                mf_index, edge, y_outside, trap_Y_outside = get_coord_trap_outside_structure(simul, axis_name)
                print("y_min", y_min)
                ax[0].set_ylim([-2,trap_Y_outside.max()])
                if y_min is not None:
                    omegay = get_trapfreq(simul,axis_name)
                    trap_1D_X = np.squeeze(np.real(simul.compute_potential_1D(axis1_name,y_min + edge,coord2)))[:,mf_shift]
                    omegax = get_trapfreq(simul,axis1_name)
                    trap_1D_Z = np.squeeze(np.real(simul.compute_potential_1D(axis2_name,coord1, y_min + edge)))[:,mf_shift]
                    omegaz = get_trapfreq(simul,axis2_name)
                    lx.set_ydata(trap_1D_X)
                    lz.set_ydata(trap_1D_Z)
                    ax[1].set_ylim([trap_1D_X.min(),trap_1D_X.max()])
                    ax[2].set_ylim([trap_1D_Z.min(),trap_1D_Z.max()])
                    ax[0].set_ylim([trap_depth,trap_Y_outside.max()])
                    
    
                    textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, ), r'$\omega_%s =%.2f (kHz) $' % (axis1_name, omegax, ), r'$\omega_%s =%.2f (kHz) $' % (axis_name, omegay, ), r'$\omega_%s =%.2f (kHz) $' % (axis2_name, omegaz, )))
                    box.set_text(textstr)
                else :
                    textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, )))
                    box.set_text(textstr)
                ly.set_ydata(np.squeeze(np.real(trap_Y_outside)))
    
    #         def updatel(val):
    #             self.wavelength_index_blue = np.argmin(np.abs(np.array(self.wavelengths_dict[0])/pu.nm - slmbdablue.val))
    #             self.wavelength_index_red = np.argmin(np.abs(np.array(self.wavelengths_dict[1])/pu.nm - slmbdared.val))
    #             trap_1D_Y = np.real(self.simulate1D(state,f,mf,"XY","Y",zcoord,xcoord,self.wavelength_index_blue,self.wavelength_index_red))#/(pu.kB*coef_norm*pu.mK)
    #             ymin_ind, y_min, trap_depth, y_outside, trap_Y_outside, CP_at_min = self.find_min_trap_1D(np.squeeze(trap_1D_Y),"Y",edge1, edge2)
    #             print("CP at y_min is, when changing lambda :", CP_at_min)
    #             print("y_min", y_min)
    #             ax[0].set_ylim([-2,trap_Y_outside.max()])
                
    #             if y_min is not None:
    #                 trap_1D_X = np.squeeze(np.real(self.simulate1D(state,f,mf,"XY","X",zcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
    #                 trap_1D_Z = np.squeeze(np.real(self.simulate1D(state,f,mf,"YZ","Z",xcoord,y_min+edge1, self.wavelength_index_blue,self.wavelength_index_red))) + CP_at_min #/(pu.kB*coef_norm*pu.mK) + CP_at_min
    #                 lx.set_ydata(trap_1D_X)
    #                 lz.set_ydata(trap_1D_Z)
    #                 ax[1].set_ylim([trap_1D_X.min(),trap_1D_X.max()])
    #                 ax[2].set_ylim([trap_1D_Z.min(),trap_1D_Z.max()])
    #                 ax[0].set_ylim([trap_depth,trap_Y_outside.max()])
    #                 omegax = self.get_trapfreq(self.x,xcoord,trap_1D_X)
    #                 omegay = self.get_trapfreq(y_outside,y_min,trap_Y_outside)
    #                 omegaz = self.get_trapfreq(self.z,zcoord,trap_1D_Z)
    #                 textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, ), r'$\omega_x=%.2f (kHz) $' % (omegax, ), r'$\omega_y=%.2f (kHz) $' % (omegay, ), r'$\omega_z=%.2f (kHz) $' % (omegaz, )))
    #                 box.set_text(textstr)
    #             else :
    #                 textstr = '\n'.join((r'$\mathrm{depth}=%.2f (mK) $' % (trap_depth, )))
    #                 box.set_text(textstr)
    # #                    print("trap X at y_min :", trap_1D_X[0])
    # #                    print("trap Z at y_min :", trap_1D_Z[33])
    # #                print("trap Y at y_min :", trap_Y_outside[ymin_ind])
    # #                print("ymin_ind",ymin_ind)
    #             ly.set_ydata(np.squeeze(np.real(trap_Y_outside)))
       
            for slider in slider_ax :
                slider.on_changed(updateP)
            plt.show()
    #         slmbdared.on_changed(updatel)
    #         slmbdablue.on_changed(updatel)
            return fig, ax, slider_ax #, slmbdablue, slmbdared
    
    def get_min_trap(simul,axis,mf=0,edge_no_surface=None):
        """Finds the minimum of the trap (ie total_potential()) computed in the simulation object
    
        Args:
            simul (Simulation object): A simulation object with computation of 1D potential already run.
            axis (str): axis along which we are looking at the trap.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.
        
        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.
            
        Returns:
            (tuple): containing:
                
                - int: Index of the position of the minimum (from the outside coordinate, putting the surface at 0).
                - float: Position of the trap minimum when putting the surface at 0.
                - float: Trap depth (ie, value of the trap at the minimum).
                - float: Height of the potential barrier for the atoms (ie difference between the trap depth and the closest local maxima).
        """
        
        if np.ndim(simul.total_potential()) >= 3:
            raise TypeError("This method can only be used if a 1D computation of the potential has been done.")
        
        mf_index, edge, y_outside, trap_outside = get_coord_trap_outside_structure(simul,axis,mf,edge_no_surface)
    
        local_minima = find_peaks(-trap_outside,distance = 10, prominence = 5e-4)
        if len(local_minima[0])==0:
            print("No local minimum found")
            return None, None, 0,0
        elif len(local_minima[0])==1 and local_minima[0][0] > 5:
            print("One local miminum found at %s" %(y_outside[local_minima[0][0]]))
            return local_minima[0][0], y_outside[local_minima[0][0]],trap_outside[local_minima[0][0]], -local_minima[1]["prominences"][0]
        elif len(local_minima[0])==1 and local_minima[0][0] <= 5:
            print('One local minimum found but too close to the edge of the structure')
            return None, None,0,0
        else :
            print('Many local minima found, taking only the lowest one into account')
            arg = np.argmin(np.real(trap_outside[local_minima[0]]))
            return local_minima[0][arg], y_outside[local_minima[0][arg]],trap_outside[local_minima[0][arg]], -local_minima[1]["prominences"][arg]
    
    def get_trapfreq(simul,axis,mf=0,edge_no_surface=None):
        """Finds the value of the trapping frequency (in Hz) along the specified axis 
    
        Args:
            simul (Simulation object): A simulation object with computation of 1D potential already run.
            axis (str): axis along which we want to compute the trapping frequency.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.
        
        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.
            
        Returns:
            float: Trapping frequency along the axis (in Hz)
        """
        if np.ndim(simul.total_potential()) >= 3:
            raise TypeError("This method can only be used if a 1D computation of the potential has been done.")
        
        mf_index, edge, y_outside, trap_outside = get_coord_trap_outside_structure(simul,axis,mf,edge_no_surface)
        trap_pos_index, trap_pos ,_,_ = get_min_trap(simul,axis,mf,edge_no_surface)
        if trap_pos is None :
            raise ValueError("Cannot compute trapping frequency if there is no local minimum.")
        
        print(y_outside.shape)
        
        print(trap_outside.shape)
        try :
            fit = np.polyfit(y_outside[5:], trap_outside[5:], 40)
            pass
        except np.linalg.LinAlgError:
            fit = np.polyfit(y_outside[5:], trap_outside[5:], 20)
            
        p = np.poly1d(fit)
        der_fit = np.real(np.gradient(p(y_outside),y_outside))
        der2_fit = np.gradient(der_fit,y_outside)
        index_min = np.argmin(np.abs(y_outside-trap_pos))
        moment2 = der2_fit[index_min]
        trap_freq = np.sqrt((moment2*kB*mK)/(simul.atomicsystem.atom.mass))*(1/(2*np.pi))
        return trap_freq
    
    def get_coord_trap_outside_structure(simul,axis,mf=0,edge_no_surface=None):
        """Returns the truncation of both the specified axis and the trap along that direction, setting 0 for the coordinatate at the edge of the structure.
    
        Args:
            simul (Simulation object): A simulation object with computation of 1D potential already run.
            axis (str): axis along which we are looking at the trap.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.
        
        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.
            
        Returns:
            (tuple): containing:
                
                - int: Index of the specified mf state in the array
                - float: Position of the edge of the structure (taken either from the Surface object or given by the user).
                - array: New coordinates, with 0 at the edge of the structure and negative values truncated.
                - array: Corresponding truncation of the trapping potential.
        """
        if np.ndim(simul.total_potential()) >= 3:
            raise TypeError("This method can only be used if a 1D computation of the potential has been done.")
        
        _, mf = check_mf(simul.atomicsystem.f,mf)
        mf_index = int(mf + simul.atomicsystem.f)    
        coord = set_axis_from_axis(axis, simul)
        
        if type(simul.surface).__name__ == "NoSurface":
            if edge_no_surface is None :
                raise ValueError("No surface for CP interactions have been specified. To restrict the search for the minimum in the right zone, you have to specify an edge")
            edge = edge_no_surface
            index_edge = np.argmin(np.abs(coord-edge))
        else:
            index_edge = np.argmin(simul.CP[:,mf_index])
            edge = coord[index_edge-1]
            
        y_outside = coord[index_edge:] - edge
        trap_outside = np.real(simul.total_potential()[index_edge:,mf_index])
        return mf_index, edge, y_outside, trap_outside


def simpleplot(coord,maskred,maskblue,trap):
    fig,ax = plt.subplots()
    plt.plot(coord*1e9,maskred, "red", linewidth = 4)   
    plt.plot(coord*1e9,maskblue,"blue",linewidth = 4)
    plt.plot(coord*1e9,trap,"darkgreen",linewidth = 5)
    plt.axhline(y=0, color='black', linestyle='--', linewidth = 2)
    plt.rc('font', size=20)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('figure', titlesize=18)
    plt.setp(ax.spines.values(), linewidth=2)
    plt.ylim([-6,6])
    plt.xlim([-10,600])
    plt.title("Trapping potential along y")
    plt.ylabel("E (mK)")
    plt.xlabel("y (nm)")


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        self.allowed_vals = kwargs.pop('allowed_vals',None)
        self.previous_val = kwargs['valinit']
        Slider.__init__(self, *args, **kwargs)
        if self.allowed_vals.all() ==None:
            self.allowed_vals = [self.valmin,self.valmax]
        for k in range(len(self.allowed_vals)):
            if self.orientation == 'vertical':
                self.hline = self.ax.axhline(self.allowed_vals[k], 0, 1, color='r', lw=1)
            else:
                self.vline = self.ax.axvline(self.allowed_vals[k], 0, 1, color='r', lw=1)

    def set_val(self, val):
        discrete_val = self.allowed_vals[abs(val-self.allowed_vals).argmin()] 
        val = discrete_val
        xy = self.poly.xy
        if self.orientation == 'vertical':
            xy[1] = 0, val
            xy[2] = 1, val
        else:
            xy[2] = val, 1
            xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(val)