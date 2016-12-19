from .source import Source
import numpy

adddiag = lambda A, v: numpy.einsum( '...ii->...i', A ).__iadd__( v )
_ = numpy.newaxis


class Nucleus( Source ):

  def __init__( self, depth ):
    self.depth = depth

  def displacement( self, xyz, poisson ):
    alpha = .5 / (1-poisson)
    z = xyz[...,2]
    xyd = xyz.copy()
    xyd[...,2] = self.depth - z
    u = self.uB( xyd, alpha )
    if z.any():
      xyp = xyz.copy()
      xyp[...,2] += self.depth
      u -= self.uA( xyp, alpha )
      u += self.uA( xyd, alpha )
      u += self.uC( xyd, alpha ) * z[...,_]
    return u

  def gradient( self, xyz, poisson ):
    alpha = .5 / (1-poisson)
    z = xyz[...,2]
    xyd = xyz.copy()
    xyd[...,2] = self.depth - z
    du = self.duB( xyd, alpha )
    du[...,2] += self.uC( xyd, alpha )
    if z.any():
      xyp = xyz.copy()
      xyp[...,2] += self.depth
      du -= self.duA( xyp, alpha ) * [1,1,-1]
      du += self.duA( xyd, alpha )
      du += self.duC( xyd, alpha ) * z[...,_,_]
    else:
      du[...,2] += self.duA( xyd, alpha )[...,2] * 2
    return du


class MogiSource( Nucleus ):

  def uA( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )
    uA = xyd * R[...,_]**-3
    uA *= .5 * alpha - .5
    return uA

  def uB( self, xyd, alpha ):
    return self.uA( xyd, alpha ) * (-2/alpha)

  def uC( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )
    uC = 3 * xyd * xyd[...,2,_] * R[...,_]**-5
    uC[...,2] *= -1
    uC[...,2] += R**-3
    uC *= 1 - alpha
    return uC

  def duA( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )
    duA = 3 * xyd[...,_,:] * xyd[...,_] * R[...,_,_]**-5
    adddiag( duA, -R[...,_]**-3 )
    duA[...,2] *= -1
    duA *= .5 - .5 * alpha
    return duA

  def duB( self, xyd, alpha ):
    return self.duA( xyd, alpha ) * (-2/alpha)

  def duC( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )
    duC = -30 * xyd[...,_,:] * xyd[...,2,_,_] * xyd[...,:,_] * R[...,_,_]**-7
    duC[...,:,2] += 6 * xyd * R[...,_]**-5
    duC[...,2,:] *= -1
    duC[...,2,:] -= 6 * xyd * R[...,_]**-5
    duC[...,2] *= -1
    adddiag( duC, 6 * xyd[...,_,2] * R[...,_]**-5 )
    duC *= .5 - .5 * alpha
    return duC


class TensileSource( Nucleus ):

  def __init__( self, depth, dip=90 ):
    rad = dip * numpy.pi / 180
    self.cosd = numpy.cos( rad )
    self.sind = numpy.sin( rad )
    self.cos2d = numpy.cos( 2*rad )
    self.sin2d = numpy.sin( 2*rad )
    self.rotmat = numpy.array([[1,0,0],[0,self.cosd,-self.sind],[0,self.sind,self.cosd]])
    self.rotmat2 = numpy.array([[1,0,0],[0,self.cos2d,-self.sin2d],[0,self.sin2d,self.cos2d]])
    Nucleus.__init__( self, depth )

  def uA( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd , axis=-1 )
    q = numpy.einsum( '...i,i->...', xyd, self.rotmat[:,2] )
    uA = numpy.einsum( '...i,ij->...j', xyd, self.rotmat2 )
    uA[...,2] *= -1
    uA *= .5 * (1-alpha) * R[...,_]**-3
    uA -= ( 1.5 * (alpha*q[...,_]**2) * R[...,_]**-5 ) * xyd
    return uA

  def uB( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd , axis=-1 )
    q = numpy.einsum( '...i,i->...', xyd, self.rotmat[:,2] )
    d = xyd[...,2]
    v = R * (R+d)
    uB1 = xyd.copy()
    uB1[...,2] = self.depth
    uB1 *= (3*q[...,_]**2) * R[...,_]**-5
    uB2 = numpy.empty_like( xyd, dtype=float )
    uB2[...,0] = xyd[...,0] * ( d * (2+d/R) / v**2 + xyd[...,1]**2 * (3*R+d) / v**3 )
    uB2[...,1] = xyd[...,1] * ( R / v**2 - xyd[...,0]**2 * (3*R+d) / v**3 )
    uB2[...,2] = ( v - xyd[...,0]**2 * (2+d/R) ) / v**2
    uB2 *= (1-1./alpha) * self.sind**2
    return uB1 + uB2

  def uC( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd , axis=-1 )
    q = numpy.einsum( '...i,i->...', xyd, self.rotmat[:,2] )
    uC1 = numpy.empty_like( xyd, dtype=float )
    uC1[...,:2] = xyd[...,:2] * numpy.einsum( '...i,i->...', xyd, -self.rotmat2[:,2] )[...,_]
    uC1[...,2] = xyd[...,0]**2 * self.sind**2
    uC1 *= -3 * R[...,_]**-5
    uC1[...,1] += self.sin2d * R**-3
    uC1[...,2] -= self.cosd**2 * R**-3
    uC1 *= 1-alpha
    uC2 = numpy.einsum( '...i,ij->...j', xyd, self.rotmat2 )
    uC2 += ( 5 * xyd * q[...,_]**2 * R[...,_]**-2 - xyd ) * [1,1,-1]
    uC2 *= alpha*3*self.depth * R[...,_]**-5
    uC3 = xyd * [-1,-1,1]
    uC3 *= alpha*3*(self.depth-xyd[...,2,_]) * R[...,_]**-5
    return uC1 + uC2 + uC3
  
  def duA( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )
    a2 = xyd[...,1] * self.sind - xyd[...,2] * self.cosd
    duA1dxyz = -3 * xyd[...,_,:] * numpy.einsum( '...i,ij->...j', xyd, self.rotmat2 )[...,:,_] * R[...,_,_]**-2
    duA1dxyz[...,2] *= -1
    duA1dxyz[...,2,:] *= -1
    duA1dxyz += self.rotmat2
    duA1dxyz *= .5 * (1-alpha) * R[...,_,_]**-3
    duA2dxyz = 5 * xyd[...,:,_] * xyd[...,_,:] * R[...,_,_]**-2
    adddiag( duA2dxyz, -1 )
    duA2dxyz[...,2] *= -1
    duA2dxyz *= a2[...,_,_]
    duA2dxyz[...,1] -= 2 * self.sind * xyd 
    duA2dxyz[...,2] -= 2 * self.cosd * xyd
    duA2dxyz *= 1.5 * alpha * a2[...,_,_] * R[...,_,_]**-5
    return duA1dxyz + duA2dxyz
    
  def duB( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )
    q = xyd[...,1]*self.sind - xyd[...,2]*self.cosd
    A3 = 1 - 3*xyd[...,0]**2 * R**-2
    d = xyd[...,2]
    J1 = -3*xyd[...,0]*xyd[...,1] * ( (3*R + d) / (R**3 * (R+d)**3) - xyd[...,0]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3./(R * (R+d)**2) + 3*xyd[...,0]**2*xyd[...,1]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J3 = A3/R**3 - J2
    K1 =-xyd[...,1] * ( (2.*R+d) / (R**3*(R+d)**2) - xyd[...,0]**2*(8.0*R**2+9.*R*d+3.0*d**2) / (R**5*(R+d)**3) ) 
    K2 =-xyd[...,0] * ( (2.*R+d) / (R**3*(R+d)**2) - xyd[...,1]**2*(8.0*R**2+9.*R*d+3.0*d**2) / (R**5*(R+d)**3) )  
    K3 =-3*xyd[...,0]*d/R**5 - K2
    W = self.sind + self.sind - 5*xyd[...,1]*q / R**2
    Wpri = self.cosd + self.cosd + 5*d*q / R**2
    a3 = 3.*q**2 / R**5        
    a4 = (1-alpha)*self.sind**2/alpha
    b3 = 3.*q*W / R**5
    c3 = 3.*q[...,_]*Wpri[...,_] / R[...,_]**5
    xyp = xyd.copy()
    xyp[...,2] = self.depth
    duBdxyz = numpy.empty( xyd.shape+(3,), dtype=float )
    duBdxyz[...,0] = a3[...,_] * -5*xyd[...,0,_]*xyp/R[...,_]**2
    duBdxyz[...,1] = b3[...,_] * xyp
    duBdxyz[...,2] = c3[...,0,_] * xyp
    duBdxyz[...,0,0] += a3
    duBdxyz[...,1,1] += a3
    duBdxyz[...,0,0] -= a4 * J3
    duBdxyz[...,0,1] -= a4 * J1
    duBdxyz[...,0,2] += a4 * K3
    duBdxyz[...,1,0] -= a4 * J1
    duBdxyz[...,1,1] -= a4 * J2
    duBdxyz[...,1,2] += a4 * K1
    duBdxyz[...,2,0] -= a4 * K3
    duBdxyz[...,2,1] -= a4 * K1
    duBdxyz[...,2,2] -= a4 * A3/R**3
    return duBdxyz
    
  def duC( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )
    q = xyd[...,1]*self.sind - xyd[...,2]*self.cosd
    p = xyd[...,1]*self.cosd + xyd[...,2]*self.sind
    s = p*self.sind + q*self.cosd
    t = p*self.cosd - q*self.sind
    A3 = 1 - 3*xyd[...,0]**2 / R**2   
    A5 = 1 - 5*xyd[...,0]**2 / R**2
    A7 = 1 - 7*xyd[...,0]**2 / R**2
    B5 = 1-5 * xyd[...,1]**2 / R**2
    B7 = 1-7 * xyd[...,1]**2 / R**2
    C5 = 1-5 * xyd[...,2]**2 / R**2
    C7 = 1-7 * xyd[...,2]**2 / R**2
    d = xyd[...,2]
    W = self.sind + self.sind - 5*xyd[...,1]*q / R**2
    Wpri = self.cosd + self.cosd + 5*d*q / R**2
    a5 = (1-alpha)*3. / R**5
    a6 = alpha*15.*self.depth/R**7
    a7 = alpha*3.*(self.depth-xyd[...,2]) / R**5
    b5 = alpha*3.*self.depth/R**5
    c6 = alpha*3./R**5
    duCdxyz = numpy.empty( xyd.shape+(3,), dtype=float )
    duCdxyz[...,0,0] = -s*A5
    duCdxyz[...,0,1] = -xyd[...,0] * (self.sin2d - 5*xyd[...,1]*s/R**2)
    duCdxyz[...,0,2] = -xyd[...,0] * (self.cos2d + 5*d*s/R**2 )
    duCdxyz[...,1,0] = -xyd[...,0]*(self.sin2d - 5*xyd[...,1]*s/R**2)
    duCdxyz[...,1,1] = -(2*xyd[...,1]*self.sin2d + s*B5)
    duCdxyz[...,1,2] = ( d*B5*self.sin2d - xyd[...,1]*C5*self.cos2d )
    duCdxyz[...,2,0] = xyd[...,0]*(1 - (2+A5)*self.sind**2)
    duCdxyz[...,2,1] = xyd[...,1] * (1 - A5*self.sind**2)
    duCdxyz[...,2,2] = -d * (1 - A5*self.sind**2)
    duCdxyz *= a5[...,_,_]
    duCdxyz[...,0,0] += a6 * q**2*A7 + a7 * -A5
    duCdxyz[...,0,1] += b5 * -5*xyd[...,0] * (t - xyd[...,1] + 7.*xyd[...,1]*q**2/R**2) / R**2 + a7 * 5*xyd[...,0]*xyd[...,1]/R**2
    duCdxyz[...,0,2] += b5 * 5*xyd[...,0] * (s - d + 7*d*q**2/R**2)/R**2 - c6 * xyd[...,0]*(1 + 5*d*(self.depth-xyd[...,2])/R**2)
    duCdxyz[...,1,0] += a6 * -xyd[...,0] * (t - xyd[...,1] + 7*xyd[...,1]*q**2/R**2) + a7 * 5*xyd[...,0]*xyd[...,1] / R**2
    duCdxyz[...,1,1] += b5 * -(2*self.sind**2 + 10*xyd[...,1]*(t-xyd[...,1])/R**2 - 5*q**2*B7/R**2) + a7 * -B5
    duCdxyz[...,1,2] += b5 * ( (3+A5)*self.sin2d - 5*xyd[...,1]*d*(2-7*q**2/R**2)/R**2 ) - c6 * xyd[...,1]*(1+5*d*(self.depth-xyd[...,2])/R**2)
    duCdxyz[...,2,0] += a6 * xyd[...,0] * (s - d + 7*d*q**2/R**2) + a7 * -5*xyd[...,0]*d / R**2
    duCdxyz[...,2,1] += b5 * ( (3.+A5)*self.sin2d - 5*xyd[...,1]*d*(2.-7.*q**2/R**2)/R**2 ) + a7 * -5*xyd[...,1]*d/R**2
    duCdxyz[...,2,2] += b5 * -(self.cos2d + 10*d*(s-d)/R**2 - 5*q**2*C7/R**2) - c6 * (self.depth-xyd[...,2])*(1+C5)
    return duCdxyz
    

class DipSource( Nucleus ):

  def __init__( self, depth, dip=90 ):
    rad = dip * numpy.pi / 180        
    self.cosd  = numpy.cos(rad)
    self.cos2d = numpy.cos(2*rad)
    self.sind  = numpy.sin(rad)
    self.sin2d = numpy.sin(2*rad)
    Nucleus.__init__( self, depth )

  def uA( self, xyd, alpha ):
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    R = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    s = p*self.sind + q*self.cosd
    t = p*self.cosd - q*self.sind
    a1 = 0.5 * ( 1 - alpha ) / R**3
    a2 = 1.5 * ( alpha*q*p ) / R**5
    uA = a1 * numpy.concatenate( [numpy.zeros_like(s), s, -t], axis=-1 ) +  a2 * xyd
    return uA

  def uB( self, xyd, alpha ):
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    R = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    a3 = ( 3.0*p*q ) / R**5
    a4 = ( 1-alpha ) * self.sind * self.cosd / alpha
    I1 =  xyd[...,1,_] * ( ( 1. / (R * (R+xyd[...,2,_])**2) ) - xyd[...,0,_]**2*( (3.0*R+xyd[...,2,_]) / (R**3 * (R+xyd[...,2,_])**3) ) )
    I2 =  xyd[...,0,_] * ( ( 1. / (R*(R+xyd[...,2,_])**2) ) - xyd[...,1,_]**2*( (3.0*R+xyd[...,2,_])/(R**3 *(R+xyd[...,2,_])**3) ) )
    I3 =  xyd[...,0,_] / R**3 - I2
    I5 =  1. / ( R*(R+xyd[...,2,_]) ) - xyd[...,0,_]**2 * ( 2*R+xyd[...,2,_] ) / ( R**3 * (R+xyd[...,2,_])**2 )
    uB = -a3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) +  a4 * numpy.concatenate( [I3, I1, I5], axis=-1 )
    return uB

  def uC( self, xyd, alpha ):
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    R = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    s = p*self.sind + q*self.cosd
    t = p*self.cosd - q*self.sind
    a5 = ( 1-alpha ) / R**3
    a6 = alpha*3*c / R**5
    A3 = 1.0 - 3.0*xyd[...,0,_]**2 / R**2
    uC = a5 * numpy.concatenate( [3*xyd[...,0,_]*t / R**2, -(self.cos2d - 3*xyd[...,1,_]*t / R**2), -A3*self.sind *self.cosd], axis=-1 )
    uC += a6 * numpy.concatenate( [-5*xyd[...,0,_]*p*q/R**2, s-5*xyd[...,1,_]*p*q/R**2, t+5*d*p*q/R**2], axis=-1 )
    return uC
    
  def duA( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    s = p*self.sind + q*self.cosd
    t = p*self.cosd - q*self.sind
    A5 = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    d = xyd[...,2,_]
    V = s - 5.*xyd[...,1,_]*p*q / R**2
    Vpri = t + 5.0*d*p*q/R**2
    a1 = 1.5 * (1 - alpha) * xyd[...,0,_] / R**5
    a2 = 1.5 * alpha*p*q / R**5
    b1 = 0.5 * (1-alpha) / R**3 
    b2 = 1.5 * alpha*V / R**5
    c2 = 1.5*alpha*Vpri / R**5
    duAdx = a1 * numpy.concatenate( [ numpy.zeros_like(s) , -s, t ], axis=-1 )
    duAdx += a2 * numpy.concatenate( [ A5 , -5*xyd[...,0,_]*xyd[...,1,_]/R**2 , -5*xyd[...,0,_]*d/R**2 ], axis=-1 )
    duAdy = b1 * numpy.concatenate( [ numpy.zeros_like(s) , self.sin2d - 3.*xyd[...,1,_]*s / R**2 , -(self.cos2d - 3.*xyd[...,1,_]*t / R**2) ], axis=-1)
    duAdy += b2 * xyd
    duAdy += a2 * [ 0, 1, 0] 
    duAdz = b1 * numpy.concatenate( [ numpy.zeros_like(s) , self.cos2d + 3*d*s/R**2 , self.sin2d - 3*d*t/R**2 ], axis=-1 )
    duAdz += c2 * xyd
    duAdz -= a2 * [ 0, 0, 1 ]
    return numpy.array( [ duAdx.T, duAdy.T, duAdz.T ]).T

  def duB( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    s = p*self.sind + q*self.cosd
    t = p*self.cosd - q*self.sind
    A3 = 1.0 - 3.0*xyd[...,0,_]**2 / R**2   
    A5 = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    J1 =-3.0*xyd[...,0,_]*xyd[...,1,_] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0,_]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0,_]**2*xyd[...,1,_]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J3 = A3/R**3 - J2
    K1 =-xyd[...,1,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,0,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) ) 
    K2 =-xyd[...,0,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,1,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) )  
    K3 =-3.*xyd[...,0,_]*d/R**5 - K2
    V = s - 5.*xyd[...,1,_]*p*q / R**2
    Vpri = t + 5.0*d*p*q/R**2
    a3 = 3.*p*q / R**5
    a4 = (1-alpha)*self.sind*self.cosd / alpha
    b3 = 3.*V / R**5
    c3 = 3*Vpri / R**5
    duBdx = a3 * numpy.concatenate( [ -A5 , 5*xyd[...,0,_]*xyd[...,1,_]/R**2 , 5*c*xyd[...,0,_]/R**2 ], axis=-1 )
    duBdx += a4 * numpy.concatenate( [ J3, J1, K3], axis=-1 )
    duBdy = -b3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c], axis=-1 )
    duBdy -= a3 * [ 0, 1, 0 ]
    duBdy += a4 * numpy.concatenate( [ J1, J2, K1 ], axis=-1 )    
    duBdz = -c3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 )
    duBdz += a4 * numpy.concatenate( [ -K3 , -K1 , A3/R**3 ], axis=-1 )
    return numpy.array( [ duBdx.T, duBdy.T, duBdz.T ]).T

  def duC( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    s = p*self.sind + q*self.cosd
    t = p*self.cosd - q*self.sind
    A5 = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    A7 = 1.0 - 7.0*xyd[...,0,_]**2 / R**2
    B5 = 1-5. * xyd[...,1,_]**2 / R**2
    B7 = 1-7. * xyd[...,1,_]**2 / R**2
    C5 = 1-5. * xyd[...,2,_]**2 / R**2
    C7 = 1-7. * xyd[...,2,_]**2 / R**2
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    a5 = (1-alpha)*3. / R**5
    a6 = alpha*15.*c / R**7
    b6 = alpha*3.*c / R**5
    duCdx = a5 * numpy.concatenate( [ t*A5 , xyd[...,0,_] * (self.cos2d - 5*xyd[...,1,_]*t/R**2) , xyd[...,0,_] * (2+A5) * self.sind*self.cosd ], axis=-1 )
    duCdx -= a6 * numpy.concatenate( [ p*q*A7 , xyd[...,0,_] * (s - 7*xyd[...,1,_]*p*q / R**2) , xyd[...,0,_] * (t + 7*d*p*q/R**2)], axis=-1 )
    duCdy = a5 * numpy.concatenate( [ xyd[...,0,_] * (self.cos2d - 5*xyd[...,1,_]*t / R**2) , 2*xyd[...,1,_]*self.cos2d + t*B5 , xyd[...,1,_]*A5*self.sind*self.cosd ], axis=-1 )
    duCdy += b6 * numpy.concatenate( [-(5*xyd[...,0,_]/R**2) * (s - 7.*xyd[...,1,_]*p*q / R**2) , self.sin2d - 10.*xyd[...,1,_]*s / R**2 - 5.*p*q*B7 / R**2 , -((3+A5) * self.cos2d + 35.*xyd[...,1,_]*d*p*q / R**4) ], axis=-1 )
    duCdz = -a5 * numpy.concatenate( [ xyd[...,0,_] * (self.sin2d - 5*d*t/R**2) , d*B5*self.cos2d + xyd[...,1,_]*C5*self.sin2d, d*A5*self.sind*self.cosd ], axis=-1 )
    duCdz -= b6 * numpy.concatenate( [ (5*xyd[...,0,_]/R**2) * (t + 7*d*p*q/R**2) , (3+A5) * self.cos2d + 35.*xyd[...,1,_]*d*p*q/R**4 , self.sin2d - 10*d*t/R**2 + 5*p*q*C7/R**2 ], axis=-1 )
    return numpy.array( [ duCdx.T, duCdy.T, duCdz.T ]).T


class StrikeSource( Nucleus ):

  def __init__( self, depth, dip=90 ):
    rad = dip * numpy.pi / 180
    self.cosd = numpy.cos(rad)
    self.sind = numpy.sin(rad)
    Nucleus.__init__( self, depth )

  def uA( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    a1 = 0.5 * ( 1 - alpha ) / R**3
    a2 = 1.5 * ( alpha*xyd[...,0,_]*q ) / R**5
    uA = a1 * numpy.concatenate( [ q , xyd[...,0,_]*self.sind , -xyd[...,0,_]*self.cosd ], axis=-1 )
    uA += a2 * xyd
    return uA

  def uB( self, xyd, alpha ):
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    R = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    a3 = ( 3.0*xyd[...,0,_]*q ) / R**5
    a4 = ( 1 - alpha )*self.sind / alpha
    I1 =  xyd[...,1,_] * ( ( 1. / (R * (R+xyd[...,2,_])**2) ) - xyd[...,0,_]**2*( (3.0*R+xyd[...,2,_]) / (R**3 * (R+xyd[...,2,_])**3) ) )
    I2 =  xyd[...,0,_] * ( ( 1. / (R*(R+xyd[...,2,_])**2) ) - xyd[...,1,_]**2*( (3.0*R+xyd[...,2,_])/(R**3 *(R+xyd[...,2,_])**3) ) )
    I4 = -xyd[...,0,_] * xyd[...,1,_] * ( 2*R+xyd[...,2,_] ) / ( R**3 * (R+xyd[...,2,_])**2 )
    uB = -a3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 )
    uB -= a4 * numpy.concatenate( [ I1, I2, I4 ], axis=-1 )   
    return uB

  def uC( self, xyd, alpha ):
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    R = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    a5 = ( 1 - alpha ) / R**5 
    a6 = alpha*3.0*c / R**5 
    uC = a5 * numpy.concatenate( [ -( R**2 - 3.0 * xyd[...,0,_]**2 ) * self.cosd , 3.0 * xyd[...,0,_] * xyd[...,1,_] * self.cosd , -3.0 * xyd[...,0,_] * xyd[...,1,_] * self.sind ], axis=-1 )
    uC += a6 * numpy.concatenate( [ q * ( 1 - (5 * xyd[...,0,_]**2 / R**2) ) , xyd[...,0,_] * ( self.sind - (5 * xyd[...,1,_] * q / R**2) ), xyd[...,0,_] * ( self.cosd + (5 * xyd[...,2,_] * q / R**2) ) ], axis=-1 )
    return uC

  def duA( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    A3 = 1.0 - 3.0*xyd[...,0,_]**2 / R**2   
    A5 = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    d = xyd[...,2,_]
    U = self.sind - 5*xyd[...,1,_]*q / R**2
    Upri = self.cosd + 5*d*q / R**2
    a1 = 0.5 * (1 - alpha) / R**3
    a2 = 1.5 * (alpha * q) / R**5
    b2 = 1.5* alpha*xyd[...,0,_]*U / R**5
    c2 = 1.5*alpha*xyd[...,0,_]*Upri / R**5
    duAdx = a1 * numpy.concatenate( [ -3.0*xyd[...,0,_]*q / R**2 , A3*self.sind , -A3*self.cosd ], axis=-1 )
    duAdx += a2 * xyd * ( A5 + [1,0,0] )
    duAdy = a1 * numpy.concatenate( [ self.sind - 3.0*xyd[...,1,_]*q / R**2 , -3.0*xyd[...,0,_]*xyd[...,1,_]*self.sind / R**2 , 3.0*xyd[...,0,_]*xyd[...,1,_]*self.cosd / R**2 ], axis=-1 )
    duAdy += b2 * xyd
    duAdy += a2 * [0,1,0] * xyd[...,0,_]
    duAdz = a1 * numpy.concatenate( [ self.cosd + 3*d*q / R**2 , 3*xyd[...,0,_]*d*self.sind / R**2 , -3*xyd[...,0,_]*d*self.cosd / R**2 ], axis=-1 )
    duAdz += c2 * xyd
    duAdz -= a2 * [0,0,1] * xyd[...,0,_]
    return numpy.array( [ duAdx.T, duAdy.T, duAdz.T ]).T

  def duB( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    A5 = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    J1 =-3.0*xyd[...,0,_]*xyd[...,1,_] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0,_]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0,_]**2*xyd[...,1,_]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J4 =-3.0*xyd[...,0,_]*xyd[...,1,_] / R**5 - J1
    K1 =-xyd[...,1,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,0,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) ) 
    K2 =-xyd[...,0,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,1,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) )  
    U = self.sind - 5*xyd[...,1,_]*q / R**2
    Upri = self.cosd + 5*d*q / R**2
    a3 = (3.0 * q) / R**5
    a4 = (1 - alpha) * self.sind / alpha
    b3 = 3*xyd[...,0,_]*U / R**5
    c3 = 3*xyd[...,0,_]*Upri / R**5
    duBdx = -a3 * ( A5 + [1,0,0] ) * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 )
    duBdx -= a4 * numpy.concatenate( [ J1, J2, K1 ], axis=-1 )
    duBdy = -b3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 )
    duBdy -= a3 * [0,1,0]* xyd[...,0,_]
    duBdy -= a4 * numpy.concatenate( [ J2, J4, K2 ], axis=-1 )
    duBdz = -c3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 )
    duBdz += a4 * numpy.concatenate( [ K1 , K2 , 3*xyd[...,0,_]*xyd[...,1,_] / R**5 ], axis=-1 ) 
    return numpy.array( [ duBdx.T, duBdy.T, duBdz.T ]).T

  def duC( self, xyd, alpha ):
    R = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    A5 = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    A7 = 1.0 - 7.0*xyd[...,0,_]**2 / R**2
    B5 = 1-5. * xyd[...,1,_]**2 / R**2
    B7 = 1-7. * xyd[...,1,_]**2 / R**2
    C7 = 1-7. * xyd[...,2,_]**2 / R**2
    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)
    a5 = 3 * (1 - alpha) / R**5 
    a6 = alpha*3.0*c / R**5 
    duCdx = a5 * ( A5 + [2,0,0] ) * numpy.concatenate( [ xyd[...,0,_]*self.cosd , xyd[...,1,_]*self.cosd , -xyd[...,1,_]*self.sind ], axis=-1 )
    duCdx += a6 * numpy.concatenate( [ -5*xyd[...,0,_]*q*(2+A7) / R**2 , A5*self.sind - 5*xyd[...,1,_]*q*A7 / R**2 , A5*self.cosd + 5*d*q*A7 / R**2 ], axis=-1 )
    duCdy = a5 * numpy.concatenate( [ xyd[...,1,_]*A5*self.cosd , xyd[...,0,_]*B5*self.cosd , -xyd[...,0,_]*B5*self.sind ], axis=-1 )
    duCdy += a6 * numpy.concatenate( [ A5*self.sind - 5*xyd[...,1,_]*q*A7 / R**2 , -5*xyd[...,0,_]*(2*xyd[...,1,_]*self.sind + q*B7) / R**2, 5*xyd[...,0,_]*(d*B7*self.sind - xyd[...,1,_]*C7*self.cosd) / R**2 ], axis=-1 )
    duCdz = a5 * numpy.concatenate( [ -d*A5*self.cosd , d*5*xyd[...,0,_]*xyd[...,1,_]*self.cosd / R**2 , -d*5*xyd[...,0,_]*xyd[...,1,_]*self.sind / R**2 ], axis=-1 )
    duCdz += a6 * numpy.concatenate( [ A5*self.cosd+5*d*q*A7 / R**2 , 5*xyd[...,0,_]*(d*B7*self.sind - xyd[...,1,_]*C7*self.cosd) / R**2, 5*xyd[...,0,_]*(2*d*self.cosd - q*C7) / R**2 ], axis=-1 )
    return numpy.array( [ duCdx.T, duCdy.T, duCdz.T ]).T
