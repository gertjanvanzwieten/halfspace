import numpy
from . import okada


def dsin( x ): return numpy.sin( x * numpy.pi / 180. )
def dcos( x ): return numpy.cos( x * numpy.pi / 180. )
def dtan( x ): return numpy.tan( x * numpy.pi / 180. )


class Source( object ):

  def strain( self, xyz, poisson ):
    grad = self.gradient( xyz, poisson )
    return .5 * ( grad + grad.swapaxes(-1,-2) )

  def stress( self, xyz, poisson, young ):
    lmbda = (poisson*young)/float((1+poisson)*(1-2*poisson))
    mu = young/float(2*(1+poisson))
    strain = self.strain( xyz, poisson )
    stress = (2*mu) * strain
    trace = lmbda * numpy.trace( strain, axis1=-2, axis2=-1 )
    stress[...,0,0] += trace
    stress[...,1,1] += trace
    stress[...,2,2] += trace
    return stress

  def __add__( self, other ):
    assert isinstance( other, Source )
    return AddSource( self, other )


class AddSource( Source ):

  def __init__( self, source1, source2 ):
    self.source1 = source1
    self.source2 = source2

  def displacement( self, xyz, poisson ):
    return self.source1.displacement( xyz, poisson ) \
         + self.source2.displacement( xyz, poisson )

  def gradient( self, xyz ):
    return self.source1.gradient( xyz, poisson ) \
         + self.source2.gradient( xyz, poisson )


class MultiFun( tuple ):

  def __call__( self, **kwargs ):
    for f in self:
      try:
        return f( **kwargs )
      except TypeError:
        pass
    raise Exception, 'not supported: ' + ', '.join( kwargs.keys() )


class OkadaSource( Source ):

  def __init__( self, **kwargs ):
    self.strike, self.dip, self.length, self.width, self.bottom, geom_args = self._rectangle( **kwargs )
    self.strikeslip, self.dipslip, self.opening, leftover = self._slip( **geom_args )
    assert not leftover, 'leftover arguments: %s' % ', '.join( leftover.keys() )
    for key, value in kwargs.items():
      numpy.testing.assert_almost_equal( value, getattr(self,key) )

    self._sourceparams = okada.SourceParams( self.strike, self.dip, self.length, self.width, self.bottom[0], self.bottom[1], self.bottom[2], self.strikeslip, self.dipslip, opening=self.opening )

  # find strike, dip, length, width, bottom
  _rectangle = MultiFun([
    lambda strike, dip, length, width, xtop, ytop, ztop, **leftover: (
      strike, dip, length, width,
      numpy.array([ xtop, ytop, ztop ]) - numpy.array([ -dcos(dip) * dcos(strike), dcos(dip) * dsin(strike), dsin(dip) ]) * width,
      leftover ),
    lambda strike, dip, length, width, bottom, **leftover: (
      strike, dip, length, width, bottom,
      leftover ),
    lambda strike, dip, length, width, xbottom, ybottom, zbottom, **leftover: (
      strike, dip, length, width,
      numpy.array([ xbottom, ybottom, zbottom ]),
      leftover ),
    lambda strike, dip, zbottom, ztop, length, xtrace, ytrace, **leftover: (
      strike, dip, length, (ztop-zbottom) / dsin(dip),
      numpy.array([ xtrace - zbottom * dcos(strike) / dtan(dip), ytrace + zbottom * dsin(strike) / dtan(dip), zbottom ]),
      leftover ),
  ])
  
  # find strikeslip, dipslip, opening
  _slip = MultiFun([
    lambda strikeslip, dipslip, opening=0., **leftover: (
      strikeslip, dipslip, opening,
      leftover ),
    lambda slip, rake, opening=0., **leftover: (
      slip * dcos(rake), slip * dsin(rake), opening,
      leftover ),
  ])

  def patches( self, n, m ):
    length = self.length / float(n)
    width = self.width / float(m)
    patches = []
    for i in range(n):
      for j in range(m):
        bottom = self.bottom + self.strikevec*((i+.5-.5*n)*length) + self.dipvec*j*width
        patch = OkadaSource(
          strike=self.strike, dip=self.dip,
          length=length, width=width,
          bottom=bottom,
          strikeslip=self.strikeslip, dipslip=self.dipslip )
        patches.append( patch )
    return patches

  @property
  def area( self ):
    return self.length * self.width

  @property
  def corners( self ):
    return self.bottom + numpy.array(
      [[ -.5 * self.length * self.strikevec,
         -.5 * self.length * self.strikevec + self.width * self.dipvec ],
       [ +.5 * self.length * self.strikevec,
         +.5 * self.length * self.strikevec + self.width * self.dipvec ]] )

  @property
  def slip( self ):
    return numpy.hypot( self.strikeslip, self.dipslip )

  @property
  def rake( self ):
    return numpy.arctan2( self.dipslip, self.strikeslip ) * 180 / numpy.pi

  @property
  def top( self ):
    return self.bottom + self.width * self.dipvec

  @property
  def xtop( self ):
    return self.top[0]

  @property
  def ytop( self ):
    return self.top[1]

  @property
  def ztop( self ):
    return self.top[2]

  @property
  def xbottom( self ):
    return self.bottom[0]

  @property
  def ybottom( self ):
    return self.bottom[1]

  @property
  def zbottom( self ):
    return self.bottom[2]

  @property
  def center( self ):
    return self.bottom + .5 * self.width * self.dipvec

  @property
  def xtrace( self ):
    return self.xbottom + self.zbottom * dcos(self.strike) / dtan(self.dip)

  @property
  def ytrace( self ):
    return self.ybottom - self.zbottom * dsin(self.strike) / dtan(self.dip)

  @property
  def dipvec( self ):
    return numpy.array( [ -dcos(self.dip) * dcos(self.strike), dcos(self.dip) * dsin(self.strike), dsin(self.dip) ] )

  @property
  def strikevec( self ):
    return numpy.array( [ dsin(self.strike), dcos(self.strike), 0. ] )

  @property
  def openvec( self ):
    return numpy.array( [ dcos(self.strike) * dsin(self.dip), -dsin(self.strike) * dsin(self.dip), dcos(self.dip) ] )

  @property
  def slipvec( self ):
    return self.strikeslip * self.strikevec + self.dipslip * self.dipvec + self.opening * self.openvec

  def displacement( self, xyz, poisson ):
    return self._sourceparams.get_displacements( xyz, poisson )

  def gradient( self, xyz, poisson ):
    return self._sourceparams.get_gradients( xyz, poisson )
