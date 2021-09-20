# Handle mouse clicks on the plot:
from nanotrappy.utils.utils import find_nearest
import numpy as np
import matplotlib.pyplot as plt


class LineSlice:
    """Allow user to drag a line on a pcolor/pcolormesh plot, and plot the Z values from that line on a separate axis.

    Example
    -------
    fig, (ax1, ax2) = plt.subplots( nrows=2 )    # one figure, two axes
    img = ax1.pcolormesh( x, y, Z )     # pcolormesh on the 1st axis
    lntr = LineSlice( img, ax2 )        # Connect the handler, plot LineSlice onto 2nd axis

    Arguments
    ---------
    img: the pcolormesh plot to extract data from and that the User's clicks will be recorded for.
    ax2: the axis on which to plot the data values from the dragged line.


    """

    def __init__(self, img, s1, s2, grid1, grid2, ax):
        """
        img: the pcolormesh instance to get data from/that user should click on
        ax: the axis to plot the line slice on
        """
        self.img = img
        self.ax = ax
        self.s1 = s1
        self.s2 = s2
        # self.data = img.get_array().reshape(img._meshWidth, img._meshHeight)
        self.data = img.get_array().reshape(s1, s2)
        self.grid1 = grid1
        self.grid2 = grid2

        # register the event handlers:
        self.cidclick = img.figure.canvas.mpl_connect("button_press_event", self)
        self.cidrelease = img.figure.canvas.mpl_connect("button_release_event", self)

        self.markers, self.arrow = None, None  # the lineslice indicators on the pcolormesh plot
        self.line = None  # the lineslice values plotted in a line

    # end __init__

    def __call__(self, event):
        """Matplotlib will run this function whenever the user triggers an event on our figure"""
        if event.inaxes != self.img.axes:
            return  # exit if clicks weren't within the `img` axes
        # if self.img.figure.canvas.manager.toolbar._active is not None:
        #     return  # exit if pyplot toolbar (zooming etc.) is active

        if event.name == "button_press_event":
            self.p1 = (event.xdata, event.ydata)  # save 1st point
        elif event.name == "button_release_event":
            self.p2 = (event.xdata, event.ydata)  # save 2nd point
            self.drawLineSlice()  # draw the Line Slice position & data

    # end __call__

    def drawLineSlice(self):
        """Draw the region along which the Line Slice will be extracted, onto the original self.img pcolormesh plot.  Also update the self.axis plot to show the line slice data."""
        """Uses code from these hints:
        http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
        http://stackoverflow.com/questions/34840366/matplotlib-pcolor-get-array-returns-flattened-array-how-to-get-2d-data-ba
        """
        self.data = self.img.get_array().reshape(self.s1, self.s2)
        x0, y0 = self.p1[0], self.p1[1]  # get user's selected coordinates
        x1, y1 = self.p2[0], self.p2[1]
        length = int(np.hypot(x1 - x0, y1 - y0))
        # x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        x = np.linspace(find_nearest(self.grid1, x0), find_nearest(self.grid2, y0), length)
        y = np.linspace(find_nearest(self.grid1, x1), find_nearest(self.grid2, y1), length)

        # Extract the values along the line with nearest-neighbor pixel value:
        # get temp. data from the pcolor plot
        zi = self.data[x.astype(np.int), y.astype(np.int)]
        print(zi)
        # Extract the values along the line, using cubic interpolation:
        # import scipy.ndimage
        # zi = scipy.ndimage.map_coordinates(self.data, np.vstack((x,y)))

        # if plots exist, delete them:
        if self.markers != None:
            if isinstance(self.markers, list):
                self.markers[0].remove()
            else:
                self.markers.remove()
        if self.arrow != None:
            self.arrow.remove()

        # plot the endpoints
        self.markers = self.img.axes.plot([x0, x1], [y0, y1], "wo")
        # plot an arrow:
        self.arrow = self.img.axes.annotate(
            "",
            xy=(x0, y0),  # start point
            xycoords="data",
            xytext=(x1, y1),  # end point
            textcoords="data",
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", color="white", alpha=0.7, linewidth=3),
        )

        # plot the data along the line on provided `ax`:
        if self.line != None:
            self.line[0].remove()  # delete the plot
        self.line = self.ax.plot(list(reversed(zi)), color="red")
        self.ax.set_ylim([-2, 2])
        self.img.figure.canvas.draw_idle()

    # end drawLineSlice()


# end class LineTrace
