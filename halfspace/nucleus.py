from .source import Source, diag
import numpy

_ = numpy.newaxis


class MogiSource( Source ):

  def __init__( self, depth ):
    self.depth = depth

  def displacement( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )

    xyd = xyz + [0,0,self.depth]
    R = numpy.linalg.norm( xyd, axis=-1 )
    uA_ = -xyd / R[...,_]**3

    xyd[...,2] -= 2 * xyz[...,2]
    R = numpy.linalg.norm( xyd, axis=-1 )
    uA = -xyd / R[...,_]**3

    uC = 6 * xyd * xyd[...,2,_] / R[...,_]**5
    uC[...,2] *= -1
    uC[...,2] += 2. / R**3

    u = uA - uA_ + xyz[...,2,_] * uC
    u *= .25 / (1-poisson)
    u -= uA
    u *= 1 - 2*poisson
    return u

  def gradient( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )

    xyd = xyz + [0,0,self.depth]
    R = numpy.linalg.norm( xyd, axis=-1 )
    R3 = R**3
    R5 = R**5
    xyd_R5 = xyd / R5[...,_]

    duAdxyz_ = 3 * xyd[...,_,:] * xyd_R5[...,:,_]
    diag(duAdxyz_)[...] -= 1. / R3[...,_]

    xyd[...,2] -= 2 * xyz[...,2]
    R = numpy.linalg.norm( xyd, axis=-1 )
    R3 = R**3
    R5 = R**5
    R7 = R**7
    xyd_R5 = xyd / R5[...,_]
    xyd_R7 = xyd / R7[...,_]

    duAdxyz = 3 * xyd[...,_,:] * xyd_R5[...,_]
    diag(duAdxyz)[...] -= 1. / R3[...,_]
    duAdxyz[...,2] *= -1

    duCdxyz = -30 * xyd[...,_,:] * xyd[...,2,_,_] * xyd_R7[...,:,_]
    duCdxyz[...,2] += 6 * xyd_R5
    duCdxyz[...,2,:] *= -1
    duCdxyz[...,2,:] -= 6 * xyd_R5
    duCdxyz[...,2] *= -1
    diag(duCdxyz)[...] += 6 * xyd_R5[...,_,2]

    uC = 6 * xyd * xyd_R5[...,_,2]
    uC[...,2] *= -1
    uC[...,2] += 2. / R3

    dudxyz = duAdxyz - duAdxyz_ + xyz[...,_,_,2] * duCdxyz
    dudxyz[...,2] += uC
    dudxyz *= .25 / (1-poisson)
    dudxyz -= duAdxyz
    dudxyz *= 1 - 2*poisson
    return dudxyz
