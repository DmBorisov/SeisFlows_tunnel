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


class regularize(custom_import('postprocess', 'base')):
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
        super(regularize, self).check()

        if 'FIXRADIUS' not in PAR:
            setattr(PAR, 'FIXRADIUS', 7.5)

        if 'LAMBDA' not in PAR:
            setattr(PAR, 'LAMBDA', 0.)


    def write_gradient(self, path):
        super(regularize, self).write_gradient(path)

        g = self.regularize(path)
        self.save(path, g, backup='noregularize')


    def process_kernels(self, path, parameters):
        """ Processes kernels in accordance with parameter settings
        """
        fullpath = path +'/'+ 'kernels'
        assert exists(path)

        #if exists(fullpath +'/'+ 'sum'):
        #    unix.mv(fullpath +'/'+ 'sum', fullpath +'/'+ 'sum_nofix')

        # mask sources and receivers
        system.run('postprocess', 'fix_near_field', 
                   hosts='all', 
                   path=fullpath)

        if PAR.SMOOTH > 0.:
            system.run('solver', 'smooth',
                       hosts='head',
                       path=path + '/' + 'kernels/sum',
                       span=PAR.SMOOTH,
                       parameters=parameters)

        system.run('solver', 'combine',
                   hosts='head',
                   path=fullpath,
                   parameters=parameters)


    def fix_near_field(self, path=''):
        """
        """
        import preprocess
        preprocess.setup()

        name = solver.check_source_names()[solver.getnode]
        fullpath = path +'/'+ name
        #print 'DB: name=', name
        #print 'DB: fullpath=', fullpath 

        g = solver.load(fullpath, suffix='_kernel')
        g_vec = solver.merge(g)
        nproc = solver.mesh.nproc

        #print 'DB: len(g_vec)=', len(g_vec)

        if not PAR.FIXRADIUS:
            return

        x,y,z = self.getcoords()
        #print 'DB: len(g)=', len(g)
        #print 'DB: len(g[vp][0])=', len(g['vp'][0])
        #print 'DB: x.shape=', x.shape
        #print 'DB: len(x)=', len(x)

        ##sys.exit("DB: stop from postporcess-regularize")

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

        #print 'DB: lx=', lx
        #print 'DB: ly=', ly
        #print 'DB: lz=', lz
        #print 'DB: nn=', nn
        #print 'DB: nx=', nx
        #print 'DB: ny=', ny
        #print 'DB: nz=', nz
        #print 'DB: dx=', dx
        #print 'DB: dy=', dy
        #print 'DB: dz=', dz


        sigma = PAR.FIXRADIUS*(dx+dz+dy)/3.0
        _, h = preprocess.load(solver.getpath +'/'+ 'traces/obs')

        # mask sources
        mask = np.exp(-0.5*((x-h.sx[0])**2.+(y-h.sy[0])**2.+(z-h.sz[0])**2.)/sigma**2.)

        # mask top
        # for matlab
        # z_sqrt=(abs(z).^(0.25)); depth_scale=1-z_sqrt/max(z_sqrt); figure; plot(depth_scale,z);
        z_factor = np.power(abs(z),0.5)
        #max_z_factor = np.amax(z_factor)
        #scale_depth = 1.0 - z_factor/max_z_factor
        #print 'DB: max(z_factor)=',max_z_factor
        #print 'DB: max(scale_depth)=',np.amax(scale_depth)
        #print 'DB: min(scale_depth)=',np.amin(scale_depth)
        #mask *= scale_depth 
        
        #mask_depth = solver.split(z)
        mask_depth = solver.split(z_factor)
        mask_d = solver.split(mask)

        ##print 'DB: sigma=',sigma
        ##print 'DB: mask=',mask 
        #print 'DB: len(mask)=', len(mask)
        #print 'DB: len(mask_d)=', len(mask_d)
        ##print 'DB: len(g)=', len(g)
        ##print 'DB: len(g)[vp][0]=', len(g['vp'][0])

        for key in solver.parameters:
            for iproc in range(nproc):
                #print 'DB: key, iproc=', key, iproc 
                #print 'DB: len(g[key][iproc])=', len(g[key][iproc])
                #print 'DB: len(mask_d[key][iproc])=', len(mask_d[key][iproc])
                weight = np.sum(mask_d['vp'][iproc]*g[key][iproc])/np.sum(mask_d['vp'][iproc])
                #print 'DB: key, iproc, weigth= ', key, iproc, weight 
                g[key][iproc] *= 1.-mask_d['vp'][iproc]
                g[key][iproc] *= mask_depth['vp'][iproc]
                #g[key][iproc] += mask_d['vp'][iproc]*weight

                #weight = np.sum(mask_d['vp'][iproc]*g[key][iproc])/np.sum(mask_d['vp'][iproc])
                ##print 'DB: key, iproc, weigth= ', key, iproc, weight 
                #g[key][iproc] *= 1.-mask_d['vp'][iproc]
                #g[key][iproc] += mask_d['vp'][iproc]*weight

        # mask receivers
        #for ir in range(h.nr):
        #    mask = np.exp(-0.5*((x-h.rx[ir])**2.+(z-h.ry[ir])**2.)/sigma**2.)
        #    for key in solver.parameters:
        #        weight = np.sum(mask*g[key][0])/np.sum(mask)
        #        g[key][0] *= 1.-mask
        #        g[key][0] += mask*weight

        solver.save(fullpath, g, suffix='_kernel')


    def regularize(self, path):
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
                    #self.nabla(m[key][iproc], g[key][iproc] , mesh, h)

        return solver.merge(g)


    def nabla(self, mesh, m, g):
        raise NotImplementedError("Must be implemented by subclass.")


    def getcoords(self):
        model_path = PATH.OUTPUT +'/'+ 'model_init'

        model = solver.load_xyz(model_path)
        nproc = solver.mesh.nproc

        #print len(model) #=5
        #print len(model['x']) #=32
        #print nproc #=32

        x = []
        y = []
        z = []

        for iproc in range(nproc):
            #print 'DB: iproc, len(model[x_loc][iproc])', iproc, len(model['x_loc'][iproc])
            x = np.append(x, model['x_loc'][iproc])
            y = np.append(y, model['y_loc'][iproc])
            z = np.append(z, model['z_loc'][iproc])

        return np.array(x), np.array(y), np.array(z)


      #if PAR.SOLVER in ['specfem2d', 'elastic2d']:
      #    from seisflows.seistools.io import loadbin
      #    x = loadbin(model_path, 0, 'x')
      #    z = loadbin(model_path, 0, 'z')
      #    mesh = stack(x, z)
      #elif PAR.SOLVER in ['specfem2d_legacy']:
      #    m = solver.load(model_path)
      #    x = g['x'][0]
      #    z = g['z'][0]
      #    mesh = stack(x, z)
      #else:
      #    msg = "postprocess.regularize can only be used with solver.specfem2d"
      #    raise Exception(msg)
