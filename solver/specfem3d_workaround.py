
import subprocess
import sys
from glob import glob
from os.path import join

import numpy as np

from seisflows.seistools.shared import getpar, setpar, Model, Minmax
from seisflows.seistools.io import loadbypar, copybin, loadbin, savebin, splitvec

from seisflows.tools import unix
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system

# added by DmBorisov
class specfem3d_workaround(custom_import('solver', 'specfem3d')):
    """ Python interface for SPECFEM3D

      See base class for method descriptions
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(specfem3d_workaround, self).check()


    def load_xyz(self, *args, **kwargs):
        """ reads SPECFEM model or kernel

          Models are stored in Fortran binary format and separated into multiple
          files according to material parameter and processor rank.
        """
        model = super(specfem3d_workaround, self).load(*args, **kwargs)
        model_path = args[0]

        model['x_loc'] = []
        model['y_loc'] = []
        model['z_loc'] = []

        for iproc in range(self.mesh.nproc):
            # read database files
            keys, vals = loadbypar(model_path, ['x_loc','y_loc','z_loc'], iproc, '', '')
            for key, val in zip(keys, vals):
                model[key] += [val]

        return model


