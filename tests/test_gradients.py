import numpy, halfspace, functools
_ = numpy.newaxis


def _ngrad( f, xyz, eps ):
  assert xyz.ndim == 2 and xyz.shape[1] == 3 # no need to make it more generic than this
  dxyz = numpy.array([.5,-.5])[:,_,_] * numpy.eye(3) * eps # pos/neg perturbations on first axis
  ngrad = numpy.subtract( *f( xyz[_,:,_,:] + dxyz[:,_,:,:] ) ) / eps # finite difference of f
  return ngrad.transpose( [0] + list(range(2,ngrad.ndim)) + [1] ) # put gradient axis last


class BaseTest( object ):

  eps = 1e-6
  xyz = numpy.array( [[2,3,-1],[3,1,-2]] ) # two arbitrary points
  xyz0 = xyz * [1,1,0] # two arbitrary points on surface

  def test_gradient( self ):
    for xyz in self.xyz0, self.xyz:
      grad = self.source.gradient( xyz, poisson=.25 )
      ngrad = _ngrad( functools.partial( self.source.displacement, poisson=.25 ), xyz, self.eps )
      numpy.testing.assert_almost_equal( ngrad, grad, decimal=7 )

  def test_divergence( self ):
    for xyz in self.xyz0, self.xyz:
      div = _ngrad( functools.partial( self.source.stress, poisson=.25, young=1e3 ), xyz, self.eps ).trace( axis1=-1, axis2=-2 )
      numpy.testing.assert_almost_equal( div, 0, decimal=6 )

  def test_traction( self ):
    stress = self.source.stress( self.xyz0, poisson=.25, young=1e6 )
    numpy.testing.assert_almost_equal( stress[...,2], 0, decimal=10 )


class TestOkada( BaseTest ):
  def __init__( self ):
    self.source = halfspace.okada(
      strike=355, dip=83,
      strikeslip=2, dipslip=.3,
      zbottom=-17, ztop=-1, length=20,
      xtrace=0, ytrace=0 )


class TestRotation( BaseTest ):
  def __init__( self ):
    self.source = halfspace.okada(
      strike=355, dip=83,
      strikeslip=2, dipslip=.3,
      zbottom=-17, ztop=-1, length=20,
      xtrace=0, ytrace=0 ).rotated( 20 )


class TestMogi( BaseTest ):
  def __init__( self ):
    self.source = halfspace.mogi( xyz=[1,2,-3] )


class TestStrike( BaseTest ):
  def __init__( self ):
    self.source = halfspace.couple( xyz=[1,2,-3], strength=[1,0,0], dip=30 )


class TestDip( BaseTest ):
  def __init__( self ):
    self.source = halfspace.couple( xyz=[1,2,-3], strength=[0,1,0], dip=30 )


class TestTensile( BaseTest ):
  def __init__( self ):
    self.source = halfspace.couple( xyz=[1,2,-3], strength=[0,0,1], dip=30 )
