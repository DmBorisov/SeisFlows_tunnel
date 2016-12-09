import sys
import numpy as np

from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.array import grid2mesh, mesh2grid, stack
from seisflows.tools.code import exists
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import
from seisflows.tools.math import nabla

# DB:
#from scipy import signal

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import solver


class regularize3d(custom_import('postprocess', 'base')):
    """ Adds regularization options to base class

        This parent class is only an abstract base class; see child classes
        TIKHONOV1, TIKHONOV1, and TOTAL_VARIATION for usable regularization.

        Prior to regularizing gradient, near field artifacts must be corrected.
        The "FIXRADIUS" parameter specifies the radius, in number of GLL points,
        within which the correction is applied.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(regularize3d, self).check()

        if 'FIXRADIUS' not in PAR:
            setattr(PAR, 'FIXRADIUS', 7.5)

        if 'LAMBDA' not in PAR:
            setattr(PAR, 'LAMBDA', 0.)


    def write_gradient(self, path):
        super(regularize3d, self).write_gradient(path)

        g = self.regularize3d(path)
        self.save(path, g, backup='noregularize')


    # modfified by DmBorisov  
    def process_kernels(self, path, parameters):
        """ Processes kernels in accordance with parameter settings
        """
        fullpath = path +'/'+ 'kernels'
        assert exists(path)

        if exists(fullpath +'/'+ 'sum'):
            unix.mv(fullpath +'/'+ 'sum', fullpath +'/'+ 'sum_nofix')

        # mask sources and receivers
        system.run('postprocess', 'fix_near_field', 
                   hosts='all', 
                   path=fullpath)

        system.run('solver', 'combine',
                   hosts='head',
                   path=fullpath,
                   parameters=parameters)

        if PAR.SMOOTH > 0.:
            system.run('solver', 'smooth',
                       hosts='head',
                       path=path + '/' + 'kernels/sum',
                       span=PAR.SMOOTH,
                       parameters=parameters)


    # modified by DmBorisov
    def fix_near_field(self, path=''):
        """
        """
        import preprocess
        preprocess.setup()

        name = solver.check_source_names()[solver.getnode]
        fullpath = path +'/'+ name

        g = solver.load(fullpath, suffix='_kernel')
        g_vec = solver.merge(g)
        nproc = solver.mesh.nproc

        if not PAR.FIXRADIUS:
            return

        x,y,z = self.getcoords()

        lx = x.max() - x.min()
        ly = y.max() - y.min()
        lz = z.max() - z.min()
        nn = x.size
        nx = np.around(np.sqrt(nn*lx/(lz*ly)))
        ny = np.around(np.sqrt(nn*ly/(lx*lz)))
        nz = np.around(np.sqrt(nn*lz/(lx*ly)))
        dx = lx/nx*1.25
        dy = ly/ny*1.25
        dz = lz/nz*1.25


        sigma = PAR.FIXRADIUS*(dx+dz+dy)/3.0
        _, h = preprocess.load(solver.getpath +'/'+ 'traces/obs')
        mask = np.exp(-0.5*((x-h.sx[0])**2.+(y-h.sy[0])**2.+(z-h.sz[0])**2.)/sigma**2.)

        scale_z = np.power(abs(z),0.5)
       
        power_win = 10
        win_x = np.power(x,power_win)
        win_y = np.power(y,power_win)
        win_z = np.power(z,power_win)

        win_x = win_x / win_x.max()
        win_y = win_y / win_y.max()
        win_z = win_z / win_z.max()

        win_x = 1.0 - win_x[::-1]
        win_y = 1.0 - win_y[::-1]
        win_z = 1.0 - win_z[::-1]

        win_x_rev = win_x[::-1]
        win_y_rev = win_y[::-1]
        win_z_rev = win_z[::-1]

        taper_x = x*0.0 + 1.0
        taper_y = y*0.0 + 1.0
        taper_z = z*0.0 + 1.0

        taper_x *= win_x
        taper_y *= win_y
        taper_z *= win_z
        taper_x *= win_x_rev
        taper_y *= win_y_rev
        taper_z *= win_z_rev

        scale_z = scale_z*taper_z + 0.1

        mask_x = solver.split(taper_x)
        mask_y = solver.split(taper_y)
        mask_z = solver.split(scale_z)
        mask_d = solver.split(mask)

        for key in solver.parameters:
            for iproc in range(nproc):
                weight = np.sum(mask_d['vp'][iproc]*g[key][iproc])/np.sum(mask_d['vp'][iproc])
                g[key][iproc] *= 1.-mask_d['vp'][iproc]
                g[key][iproc] *= mask_z['vp'][iproc]
                g[key][iproc] *= mask_x['vp'][iproc]
                g[key][iproc] *= mask_y['vp'][iproc]


        #sigma = 1.0
        ## mask receivers
        #for ir in range(h.nr):
        #    mask = np.exp(-0.5*((x-h.rx[ir])**2.+(y-h.ry[ir])**2.+(z-h.rz[ir])**2.)/sigma**2.)
        #    mask_d = solver.split(mask)
        #    #mask = np.exp(-0.5*((x-h.rx[ir])**2.+(z-h.ry[ir])**2.)/sigma**2.)
        #    for key in solver.parameters:
        #        for iproc in range(nproc):
        #            #weight = np.sum(mask*g[key][0])/np.sum(mask)
        #            g[key][iproc] *= 1.-mask_d['vp'][iproc]
        #            #g[key][0] += mask*weight

        solver.save(fullpath, g, suffix='_kernel')


    def regularize3d(self, path):
        assert (exists(path))

        g = solver.load(path +'/'+ 'gradient', suffix='_kernel')
        if not PAR.LAMBDA:
            return solver.merge(g)

        m = solver.load(path +'/'+ 'model')
        mesh = self.getmesh()

        for key in solver.parameters:            
            for iproc in range(PAR.NPROC):
                g[key][iproc] += PAR.LAMBDA *\
                    self.nabla(mesh, m[key][iproc], g[key][iproc])

        return solver.merge(g)


    def nabla(self, mesh, m, g):
        raise NotImplementedError("Must be implemented by subclass.")


    # modified by DmBorisov
    def getcoords(self):
        model_path = PATH.OUTPUT +'/'+ 'model_init'

        model = solver.load_xyz(model_path)
        nproc = solver.mesh.nproc

        x = []
        y = []
        z = []

        for iproc in range(nproc):
            x = np.append(x, model['x_loc'][iproc])
            y = np.append(y, model['y_loc'][iproc])
            z = np.append(z, model['z_loc'][iproc])

        return np.array(x), np.array(y), np.array(z)


    def tukeywin2(window_length, alpha):
        '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
        that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
        at \alpha = 0 it becomes a Hann window.
 
        We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
        output
 
        Reference
        ---------
        http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
        '''
        # Special cases
        if alpha <= 0:
            return np.ones(window_length) #rectangular window
        elif alpha >= 1:
            return np.hanning(window_length)

        # Normal case
        x = np.linspace(0, 1, window_length)
        w = np.ones(x.shape)

        # first condition 0 <= x < alpha/2
        first_condition = x<alpha/2
        w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

        # second condition already taken care of

        # third condition 1 - alpha / 2 <= x <= 1
        third_condition = x>=(1 - alpha/2)
        w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

        return w


















