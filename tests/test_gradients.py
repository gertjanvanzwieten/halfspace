import numpy, halfspace

class BaseTest( object ):

  eps = 1e-6

  def test_gradient( self ):

    x, y, z = 2, 3, -1
    
    grad = self.source.gradient( [x,y,0], poisson=.25 )
    
    dudx = ( self.source.displacement( [x+self.eps,y,0], poisson=.25 )
           - self.source.displacement( [x-self.eps,y,0], poisson=.25 ) ) / (2*self.eps)
    dudy = ( self.source.displacement( [x,y+self.eps,0], poisson=.25 )
           - self.source.displacement( [x,y-self.eps,0], poisson=.25 ) ) / (2*self.eps)
    dudz = ( self.source.displacement( [x,y,0],          poisson=.25 )
           - self.source.displacement( [x,y,-self.eps],  poisson=.25 ) ) / self.eps

    numpy.testing.assert_almost_equal( [ dudx, dudy, dudz ], grad, decimal=7 )

    grad = self.source.gradient( [x,y,z], poisson=.25 )
    
    dudx = ( self.source.displacement( [x+self.eps,y,z], poisson=.25 )
           - self.source.displacement( [x-self.eps,y,z], poisson=.25 ) ) / (2*self.eps)
    dudy = ( self.source.displacement( [x,y+self.eps,z], poisson=.25 )
           - self.source.displacement( [x,y-self.eps,z], poisson=.25 ) ) / (2*self.eps)
    dudz = ( self.source.displacement( [x,y,z+self.eps], poisson=.25 )
           - self.source.displacement( [x,y,z-self.eps], poisson=.25 ) ) / (2*self.eps)

    numpy.testing.assert_almost_equal( [ dudx, dudy, dudz ], grad, decimal=7 )

  def test_divergence( self ):

    x, y, z = 2, 3, -1

    dsxdx = ( self.source.stress( [x+self.eps,y,0], poisson=.25, young=1e6 )
            - self.source.stress( [x-self.eps,y,0], poisson=.25, young=1e6 ) )[:,0] / (2*self.eps)
    dsydy = ( self.source.stress( [x,y+self.eps,0], poisson=.25, young=1e6 )
            - self.source.stress( [x,y-self.eps,0], poisson=.25, young=1e6 ) )[:,1] / (2*self.eps)
    dszdz = ( self.source.stress( [x,y,0],          poisson=.25, young=1e6 )
            - self.source.stress( [x,y,-self.eps],  poisson=.25, young=1e6 ) )[:,2] / self.eps

    numpy.testing.assert_almost_equal( dsxdx + dsydy + dszdz, 0, decimal=5 )

    dsxdx = ( self.source.stress( [x+self.eps,y,z], poisson=.25, young=1e6 )
            - self.source.stress( [x-self.eps,y,z], poisson=.25, young=1e6 ) )[:,0] / (2*self.eps)
    dsydy = ( self.source.stress( [x,y+self.eps,z], poisson=.25, young=1e6 )
            - self.source.stress( [x,y-self.eps,z], poisson=.25, young=1e6 ) )[:,1] / (2*self.eps)
    dszdz = ( self.source.stress( [x,y,z+self.eps], poisson=.25, young=1e6 )
            - self.source.stress( [x,y,z-self.eps], poisson=.25, young=1e6 ) )[:,2] / (2*self.eps)
    
    numpy.testing.assert_almost_equal( dsxdx + dsydy + dszdz, 0, decimal=5 )

  def test_traction( self ):

    x, y = 2, 3

    stress = self.source.stress( [x,y,0], poisson=.25, young=1e6 )

    numpy.testing.assert_almost_equal( stress[:,2], 0, decimal=13 )


class TestOkada( BaseTest ):

  def __init__( self ):

    self.source = halfspace.OkadaSource(
      strike=355, dip=83,
      strikeslip=2, dipslip=.3,
      zbottom=-17e3, ztop=-1e3, length=20e3,
      xtrace=0, ytrace=0 )


class TestMogi( BaseTest ):

  def __init__( self ):

    self.source = halfspace.MogiSource( xyz=[1,2,-3] ) * 2
