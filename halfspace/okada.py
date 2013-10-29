import numpy, sys, os
from cffi import FFI

ffi = FFI()
ffi.cdef( '''
typedef struct {
  double strike, dip;
  double length, width;
  double xbottom, ybottom, zbottom;
  double strikeslip, dipslip, opening;
} OkadaSource;

void get_displacements( double *out, OkadaSource *src, double *where, double poisson, int count );
void get_gradients( double *out, OkadaSource *src, double *where, double poisson, int count );
''' )

dirname = os.path.dirname( sys.modules[__name__].__file__ )
libokadapath = os.path.join( dirname, 'libokada.so' )
libokada = ffi.dlopen( libokadapath )

def get_data( sourceparams, xyz, poisson, func, *shape ):
  xyz = numpy.ascontiguousarray( xyz, dtype=float )
  assert xyz.shape[-1] == 3
  out = numpy.empty( xyz.shape + shape )
  func(
    ffi.cast( 'double *', out.ctypes.data ),
    ffi.new( 'OkadaSource *', sourceparams ),
    ffi.cast( 'double *', xyz.ctypes.data ),
    poisson, xyz.size//3 )
  return out

def get_displacements( sourceparams, xyz, poisson ):
  return get_data( sourceparams, xyz, poisson, libokada.get_displacements )

def get_gradients( sourceparams, xyz, poisson ):
  return get_data( sourceparams, xyz, poisson, libokada.get_gradients, 3 )
