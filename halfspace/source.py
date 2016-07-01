import numpy


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
    if other is 0: # to allow sum
      return self
    assert isinstance( other, Source )
    return AddSource( self, other )

  def __radd__( self, other ):
    return self + other


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
