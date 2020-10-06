import matplotlib.pyplot as plt
import numpy as np
from datahandler import DataHandler
import matplotlib.font_manager as font_manager

def save_figure(fig, path):
    width = 5.895
    height = width / 1.618
    size = (width, height)
    fig.set_size_inches(size)
    font_dirs = ['/usr/share/fonts/' ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)

    font = {'family' : "Times New Roman",
            'size'   : 12}
    plt.rc('font', **font)
    #fig.tight_layout()
    fig.savefig(path)

def parabolic():
    dh1 = DataHandler.fromcsv("data/1d_parabolic_for_figures.csv", 'cpu', 0)

    dh2 = DataHandler.fromcsv("data/2d_parabolic_for_figures.csv", 'cpu', 0)

    dh2_drag = DataHandler.fromcsv("data/2d_parabolic_drag_for_figures.csv", 'cpu', 0)

    fig = plt.figure()
    p1, = plt.plot(np.ones(dh1.data_np[0][:,0, 0].shape), dh1.data_np[0][:,0, 0], '--', label='Free fall')
    p2, = plt.plot(dh2.data_np[0][:,0, 0], dh2.data_np[0][:,0, 1], label='Parabola')
    p3, = plt.plot(dh2_drag.data_np[0][:,0, 0], dh2_drag.data_np[0][:,0, 1], '-.', label='Parabola with drag')
    plt.xlim((0,10))
    plt.ylim((0,16))
    plt.legend(handles=(p1, p2, p3), loc='upper right')
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r"$y$ [m]")
    return fig

def harmonic():
    dh1 = DataHandler.fromcsv("data/simple_harmonic_for_figures.csv", 'cpu', 0)

    dh2 = DataHandler.fromcsv("data/damped_harmonic_for_figures.csv", 'cpu', 0)


    fig = plt.figure()
    p1, = plt.plot(dh1.data_np[0][:,0, 0], dh1.data_np[0][:,0, 1], label='Simple harmonic')
    p2, = plt.plot(dh2.data_np[0][:,0, 0], dh2.data_np[0][:,0, 1], '--', label='Damped harmonic')
    plt.xlim((-6,6))
    plt.ylim((-8,8))
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r'$\frac{\mathrm{d} x}{\mathrm{d} t}$ [ms$^{-1}]$')
    plt.legend(handles=(p1, p2,),loc='upper right')
    return fig

def lotka_volterra():
    dh1 = DataHandler.fromcsv("data/lotka_volterra_2_traj.csv", 'cpu', 0)
    fig = plt.figure()
    p1, = plt.plot(dh1.time_np[0], dh1.data_np[0][:,0, 0],  label='Prey')
    p2, = plt.plot(dh1.time_np[0], dh1.data_np[0][:,0, 1], '--', label='Predator')
    #plt.xlim((-6,6))
    #plt.ylim((-8,8))
    plt.xlabel(r"$t$")
    plt.ylabel('Population size')
    plt.legend(handles=(p1, p2,),loc='upper right')
    fig2 = plt.figure()
    p3, = plt.plot(dh1.data_np[0][:,0, 0], dh1.data_np[0][:,0, 1])
    #plt.xlim((-6,6))
    #plt.ylim((-8,8))
    plt.xlabel("Prey population")
    plt.ylabel('Predator population')
    return fig, fig2

def activation_functions():
    x = np.linspace(-10, 10, 100)
    relu = np.copy(x)
    relu[relu < 0 ] = 0
    lrelu = np.copy(x)
    lrelu[lrelu < 0 ] *= 0.05
    fig = plt.figure()
    h1 = plt.plot(x, relu, '-', linewidth=3, label='ReLU')
    h2 = plt.plot(x, lrelu, '--', linewidth=3, label='Leaky ReLU')
    plt.xlim((-10, 10))
    plt.ylim((-1, 10))
    plt.xlabel('x')
    plt.ylabel('a(x)')
    plt.legend(loc='upper left')
    return fig

if __name__=='__main__':
    fig= harmonic()
    save_figure(fig, '/home/daniel/Desktop/thesis_images/new_method/parabolic_v2.eps')

    #save_figure(fig2, '/home/daniel/Desktop/thesis_images/new_method/lotka_volterra_phase.eps')
    plt.show()