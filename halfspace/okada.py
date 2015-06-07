import numpy, sys, os, ctypes

dirname = os.path.dirname(__file__)
libokadapath = os.path.join( dirname, 'libokada.so' )
libokada = ctypes.cdll.LoadLibrary( libokadapath )

# typedef struct {
#   double strike, dip;
#   double length, width;
#   double xbottom, ybottom, zbottom;
#   double strikeslip, dipslip, opening;
# } OkadaSource;
# 
# void get_displacements( double *out, OkadaSource *src, double *where, double poisson, int count );
# void get_gradients( double *out, OkadaSource *src, double *where, double poisson, int count );

class SourceParams( ctypes.Structure ):

  _fields_ = [
    ( 'strike',     ctypes.c_double ),
    ( 'dip',        ctypes.c_double ),
    ( 'length',     ctypes.c_double ),
    ( 'width',      ctypes.c_double ),
    ( 'xbottom',    ctypes.c_double ),
    ( 'ybottom',    ctypes.c_double ),
    ( 'zbottom',    ctypes.c_double ),
    ( 'strikeslip', ctypes.c_double ),
    ( 'dipslip',    ctypes.c_double ),
    ( 'opening',    ctypes.c_double ),
  ]

  def _call( self, func, xyz, poisson, *shape ):
    xyz = numpy.ascontiguousarray( xyz, dtype=float )
    assert xyz.shape[-1] == 3
    out = numpy.empty( xyz.shape + shape, dtype=numpy.double )
    func( out.ctypes, self, xyz.ctypes, ctypes.c_double(poisson), ctypes.c_int(xyz.size//3) )
    return out
  
  def get_displacements( self, xyz, poisson ):
    return self._call( libokada.get_displacements, xyz, poisson )
  
  def get_gradients( self, xyz, poisson ):
    return self._call( libokada.get_gradients, xyz, poisson, 3 )
