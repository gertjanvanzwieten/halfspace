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
    diag = numpy.einsum( '...ii->...i', stress )
    diag += lmbda * numpy.trace( strain, axis1=-2, axis2=-1 )[...,numpy.newaxis]
    return stress

  def translated( self, xy ):
    assert len(xy) == 2
    return TranslateSource( self, xy ) if any(xy) else self

  def rotated( self, angle ):
    return RotateSource( self, angle ) if angle else self

  def __add__( self, other ):
    if other is 0: # to allow sum
      return self
    assert isinstance( other, Source )
    return AddSource( self, other )

  def __radd__( self, other ):
    return self.__add__( other )

  def __mul__( self, other ):
    assert isinstance( other, (int,float) )
    if other == 1:
      return self
    return ScaleSource( self, other )

  def __rmul__( self, other ):
    return self.__mul__( other )


class AddSource( Source ):

  def __init__( self, source1, source2 ):
    assert isinstance( source1, Source )
    assert isinstance( source2, Source )
    self.source1 = source1
    self.source2 = source2

  def displacement( self, xyz, poisson ):
    return self.source1.displacement( xyz, poisson ) \
         + self.source2.displacement( xyz, poisson )

  def gradient( self, xyz, poisson ):
    return self.source1.gradient( xyz, poisson ) \
         + self.source2.gradient( xyz, poisson )


class ScaleSource( Source ):

  def __init__( self, source, scale ):
    assert isinstance( source, Source )
    assert isinstance( scale, (int,float) )
    self.source = source
    self.scale = scale

  def displacement( self, xyz, poisson ):
    return self.scale * self.source.displacement( xyz, poisson )

  def gradient( self, xyz, poisson ):
    return self.scale * self.source.gradient( xyz, poisson )

  def __mul__( self, other ):
    return self.source.__mul__( self.scale * scale )


class TranslateSource( Source ):

  def __init__( self, source, xy ):
    assert isinstance( source, Source )
    self.source = source
    self.xy = numpy.array( xy )
    assert self.xy.shape == (2,)

  def translated( self, xy ):
    return self.source.translated( self.xy + xy )

  def _translated( self, array ):
    array = numpy.array( array, dtype=float )
    array[...,:2] -= self.xy
    return array

  def displacement( self, xyz, poisson ):
    xyz = self._translated( xyz )
    return self.source.displacement( xyz, poisson )

  def gradient( self, xyz, poisson ):
    xyz = self._translated( xyz )
    return self.source.gradient( xyz, poisson )


class RotateSource( Source ):

  def __init__( self, source, angle ):
    assert isinstance( source, Source )
    self.source = source
    self.angle = angle
    rad = angle * numpy.pi / 180
    cos = numpy.cos( rad )
    sin = numpy.sin( rad )
    self.rotmat = numpy.array([[cos,sin],[-sin,cos]])

  def rotated( self, angle ):
    return self.source.rotated( self.angle + angle )

  def _rotated( self, array ):
    array = numpy.array( array, dtype=float )
    array[...,:2] = numpy.dot( array[...,:2], self.rotmat )
    return array

  def _rotateback( self, array, axis ):
    swapped = array.swapaxes(-1,axis)
    swapped[...,:2] = numpy.dot( swapped[...,:2], self.rotmat.T )

  def displacement( self, xyz, poisson ):
    xyz = self._rotated( xyz )
    displacement = self.source.displacement( xyz, poisson )
    self._rotateback( displacement, axis=-1 )
    return displacement

  def gradient( self, xyz, poisson ):
    xyz = self._rotated( xyz )
    gradient = self.source.gradient( xyz, poisson )
    self._rotateback( gradient, axis=-1 )
    self._rotateback( gradient, axis=-2 )
    return gradient
