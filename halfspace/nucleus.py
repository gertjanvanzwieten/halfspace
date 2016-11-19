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


class TensileSource( Source ):
  def __init__( self, depth, dip=90 ):
    self.depth = depth
    rad = dip * numpy.pi / 180
    self.cosd  = numpy.cos(rad)
    self.cos2d = numpy.cos(2*rad)
    self.sind  = numpy.sin(rad)
    self.sin2d = numpy.sin(2*rad)
  
  def displacement( self, xyz, poisson ):  
    xyz = numpy.asarray( xyz )

    xyd = xyz.copy()
    xyd[...,2] += self.depth
    
    alpha = .5 / (1 - poisson)

    Rprime = numpy.linalg.norm( xyd , axis=-1 )
    qprime = xyd[...,1]*self.sind - xyd[...,2]*self.cosd  
    pprime = xyd[...,1]*self.cosd + xyd[...,2]*self.sind    
    sprime = pprime*self.sind + qprime*self.cosd
    tprime = pprime*self.cosd - qprime*self.sind

    a1prime = 0.5*(1-alpha) / Rprime**3
    a2prime = 1.5*(alpha*qprime**2) / Rprime**5  
    uAprime = a1prime[...,_] * numpy.concatenate( [xyd[...,0,_], tprime[...,_], sprime[...,_]], axis=-1 ) - a2prime[...,_] * xyd

    xyd[...,2] -= xyz[...,2] * 2

    d = xyd[...,2,_]
    c = self.depth*numpy.ones_like(d)

    R = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    q = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    s = p*self.sind + q*self.cosd
    t = p*self.cosd - q*self.sind
      
    I1 = xyd[...,1,_] * ( ( 1. / (R * (R+xyd[...,2,_])**2) ) - xyd[...,0,_]**2*( (3.0*R+xyd[...,2,_]) / (R**3 * (R+xyd[...,2,_])**3) ) )
    I2 = xyd[...,0,_] * ( ( 1. / (R*(R+xyd[...,2,_])**2) ) - xyd[...,1,_]**2*( (3.0*R+xyd[...,2,_])/(R**3 *(R+xyd[...,2,_])**3) ) )
    I3 = xyd[...,0,_] / R**3 - I2
    I5 = 1. / ( R*(R+xyd[...,2,_]) ) - xyd[...,0,_]**2 * ( 2*R+xyd[...,2,_] ) / ( R**3 * (R+xyd[...,2,_])**2 )
  
    A3 = 1.0 - 3.0*xyd[...,0,_]**2 / R**2   
    
    a1 = 0.5*(1-alpha) / R**3
    a2 = 1.5*(alpha*q**2) / R**5
    a3 = (3.0*q**2) / R**5
    a4 = (1-alpha)*self.sind**2 / alpha  
    a5 = (1-alpha) / R**3
    a6 = alpha*3.*c / R**5
    a7 = alpha*3.*xyz[...,2,_] / R**5
        
    uA = a1 * numpy.concatenate( [xyd[...,0,_], t, s], axis=-1 ) \
       - a2 * xyd
    uB = a3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) \
       - a4 * numpy.concatenate( [ I3 , I1 , I5 ], axis=-1 )
    uC = a5 * numpy.concatenate( [-3*xyd[...,0,_]*s/R**2 , self.sin2d - 3*xyd[...,1,_]*s/R**2 , -(1 - A3*self.sind**2) ], axis=-1 ) \
       + a6 * numpy.concatenate( [5*xyd[...,0,_]*q**2/R**2 , t - xyd[...,1,_] + 5*xyd[...,1,_]*q**2/R**2 , -(s - d + 5*d*q**2/R**2)], axis=-1 ) \
       + a7 * [-1,-1,1] * xyd 
            
    return uA - uAprime + uB + xyz[...,2,_]*uC 
    
    
  def gradient( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )

    xyd = xyz.copy()
    xyd[...,2] += self.depth

    alpha  = .5 / ( 1 - poisson )

    Rprime = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    qprime = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    pprime = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind    
    sprime = pprime*self.sind + qprime*self.cosd
    tprime = pprime*self.cosd - qprime*self.sind    

    A3prime = 1.0 - 3.0*xyd[...,0,_]**2 / Rprime**2   
    A5prime = 1.0 - 5.0*xyd[...,0,_]**2 / Rprime**2
    dprime = xyd[...,2,_]
    Uprime    = self.sind - 5*xyd[...,1,_]*qprime / Rprime**2
    Upriprime = self.cosd + 5*xyd[...,2,_]*qprime / Rprime**2    
    Wprime    = self.sind + Uprime
    Wpriprime = self.cosd + Upriprime

    a1prime = 0.5*(1-alpha) / Rprime**3
    a2prime = 1.5*alpha*qprime**2 / Rprime**5    
    b2prime = 1.5*alpha*qprime*Wprime / Rprime**5
    c2prime = 1.5*alpha*qprime*Wpriprime / Rprime**5

    duAdxprime = a1prime * numpy.concatenate( [ A3prime , -3.*xyd[...,0,_]*tprime/Rprime**2, -3.*xyd[...,0,_]*sprime/Rprime**2 ], axis=-1 ) \
               + a2prime * numpy.concatenate( [ -A5prime , 5.*xyd[...,0,_]*xyd[...,1,_]/Rprime**2 , 5.*xyd[...,0,_]*dprime / Rprime**2], axis=-1 )                
    duAdyprime = a1prime * numpy.concatenate( [ -3*xyd[...,0,_]*xyd[...,1,_]/Rprime**2 , self.cos2d - 3*xyd[...,1,_]*tprime/Rprime**2 , \
                                 self.sin2d - 3.*xyd[...,1,_]*sprime/Rprime**2 ], axis=-1 ) \
               - b2prime * xyd \
               - a2prime * [ 0, 1, 0 ]                                                 
    duAdzprime = a1prime * numpy.concatenate( [ 3.*xyd[...,0,_]*dprime/Rprime**2 , -(self.sin2d - 3.*dprime*tprime/Rprime**2) , \
                                           self.cos2d + 3.*dprime*sprime/Rprime**2 ], axis=-1 )\
               - c2prime * xyd \
               + a2prime * [ 0, 0 , 1] 
    
    xyd[...,2] -= xyz[...,2] * 2
    
    R      = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    q      = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    p      = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    s      = p*self.sind + q*self.cosd
    t      = p*self.cosd - q*self.sind

    A3      = 1.0 - 3.0*xyd[...,0,_]**2 / R**2   
    A5      = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    A7      = 1.0 - 7.0*xyd[...,0,_]**2 / R**2
    
    B5 = 1-5. * xyd[...,1,_]**2 / R**2
    B7 = 1-7. * xyd[...,1,_]**2 / R**2
    
    C5 = 1-5. * xyd[...,2,_]**2 / R**2
    C7 = 1-7. * xyd[...,2,_]**2 / R**2
   
    d      = xyd[...,2,_]
    c      = self.depth*numpy.ones_like(d)
    
    J1 =-3.0*xyd[...,0,_]*xyd[...,1,_] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0,_]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0,_]**2*xyd[...,1,_]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J3 = A3/R**3 - J2
    
    K1 =-xyd[...,1,_] * ( (2.*R+d) / (R**3*(R+d)**2) - xyd[...,0,_]**2*(8.0*R**2+9.*R*d+3.0*d**2) / (R**5*(R+d)**3) ) 
    K2 =-xyd[...,0,_] * ( (2.*R+d) / (R**3*(R+d)**2) - xyd[...,1,_]**2*(8.0*R**2+9.*R*d+3.0*d**2) / (R**5*(R+d)**3) )  
    K3 =-3.*xyd[...,0,_]*d/R**5 - K2
    
    U         = self.sind - 5*xyd[...,1,_]*q / R**2
    Upri      = self.cosd + 5*d*q / R**2
    
    W         = self.sind + U
    Wpri      = self.cosd + Upri
    
    a1      = 0.5*(1-alpha) / R**3
    a2      = 1.5*alpha*q**2 / R**5
    a3      = 3.*q**2 / R**5        
    a4      = (1-alpha)*self.sind**2/alpha
    a5      = (1-alpha)*3. / R**5
    a6      = alpha*15.*c/R**7
#   a7      = alpha*15.*xyd[...,0,_]*xyz[...,2,_] / R**7
    a7      = alpha*3.*xyz[...,2,_] / R**5
    
    b2      = 1.5*alpha*q*W/R**5
    b3      = 3.*q*W / R**5
    b5      = alpha*3.*c/R**5
    
    c2      = 1.5*alpha*q*Wpri / R**5
    c3      = 3.*q*Wpri / R**5
    c6      = alpha*3./R**5
    
    duAdx = a1 * numpy.concatenate( [ A3 , -3.*xyd[...,0,_]*t/R**2, -3.*xyd[...,0,_]*s/R**2 ], axis=-1 ) \
          + a2 * numpy.concatenate( [ -A5 , 5.*xyd[...,0,_]*xyd[...,1,_]/R**2 , 5.*xyd[...,0,_]*d/R**2 ], axis=-1 )
    duBdx = a3 * numpy.concatenate( [ A5 , -5*xyd[...,0,_]*xyd[...,1,_]/R**2 , -5*c*xyd[...,0,_]/R**2 ], axis=-1 ) \
          - a4 * numpy.concatenate( [ J3, J1, K3], axis=-1 )         
    duCdx = a5 * numpy.concatenate( [ -s*A5 , -xyd[...,0,_]*(self.sin2d - 5*xyd[...,1,_]*s/R**2), \
                                 xyd[...,0,_]*(1 - (2+A5)*self.sind**2) ], axis=-1 ) \
          + a6 * numpy.concatenate( [ q**2*A7 , -xyd[...,0,_] * (t - xyd[...,1,_] + 7*xyd[...,1,_]*q**2/R**2) , \
                                 xyd[...,0,_] * (s - d + 7*d*q**2/R**2) ], axis=-1 ) \
          + a7 * numpy.concatenate( [-A5 , 5*xyd[...,0,_]*xyd[...,1,_] / R**2 , -5*xyd[...,0,_]*d / R**2], axis=-1 )         
    duAdy = a1 * numpy.concatenate( [ -3*xyd[...,0,_]*xyd[...,1,_]/R**2 , self.cos2d - 3*xyd[...,1,_]*t/R**2 , \
                                 self.sin2d - 3*xyd[...,1,_]*s/R**2 ], axis=-1 ) \
          - b2 * xyd \
          - a2 * [ 0, 1, 0 ]                             
    duBdy = b3 * numpy.concatenate( [ xyd[...,0,_], xyd[...,1,_], c ], axis=-1 ) \
          + a3 * [ 0, 1, 0 ] \
          - a4 * numpy.concatenate( [ J1, J2, K1 ], axis=-1 )
                                                                                                             
    duCdy = a5 * numpy.concatenate( [-xyd[...,0,_] * (self.sin2d - 5*xyd[...,1,_]*s/R**2) , \
                                -(2*xyd[...,1,_]*self.sin2d + s*B5) , \
                                 xyd[...,1,_] * (1 - A5*self.sind**2) ], axis=-1) \
          + b5 * numpy.concatenate( [-5*xyd[...,0,_] * (t - xyd[...,1,_] + 7.*xyd[...,1,_]*q**2/R**2) / R**2 , 
                                -(2*self.sind**2 + 10*xyd[...,1,_]*(t-xyd[...,1,_])/R**2 - 5*q**2*B7/R**2) , 
                                 (3.+A5)*self.sin2d - 5*xyd[...,1,_]*d*(2.-7.*q**2/R**2)/R**2 ], axis=-1 )\
          + a7 * numpy.concatenate( [ 5*xyd[...,0,_]*xyd[...,1,_]/R**2, -B5, -5*xyd[...,1,_]*d/R**2], axis=-1 )          
    
    duAdz = a1 * numpy.concatenate( [ 3.*xyd[...,0,_]*d/R**2 , -(self.sin2d - 3.*d*t/R**2) , self.cos2d + 3.*d*s/R**2 ], axis=-1 )\
          - c2 * xyd \
          + a2 * [ 0, 0 , 1] 
    duBdz = c3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c], axis=-1 ) \
          + a4 * numpy.concatenate( [ K3, K1, -A3/R**3], axis=-1 )   
    duCdz = a5 * numpy.concatenate( [ -xyd[...,0,_] * (self.cos2d + 5*d*s/R**2) , d*B5*self.sin2d - xyd[...,1,_]*C5*self.cos2d, 
                                 -d * (1 - A5*self.sind**2) ], axis=-1 ) \
          + b5 * numpy.concatenate( [ 5*xyd[...,0,_] * (s - d + 7*d*q**2/R**2)/R**2 , \
                                 (3+A5)*self.sin2d - 5*xyd[...,1,_]*d*(2-7*q**2/R**2)/R**2 , \
                                -(self.cos2d + 10*d*(s-d)/R**2 - 5*q**2*C7/R**2) ], axis=-1) \
          - c6 * numpy.concatenate( [ xyd[...,0,_]*(1 + 5*d*xyz[...,2,_]/R**2) , xyd[...,1,_]*(1+5*d*xyz[...,2,_]/R**2), \
                                 xyz[...,2,_]*(1+C5) ], axis=-1)

    uC = a5 * numpy.concatenate( [-xyd[...,0,_]*s , (R**2/3)*(self.sin2d - 3*xyd[...,1,_]*s/R**2) , 
                             -(R**2/3)*(1 - A3*self.sind**2) ], axis=-1 ) \
       + b5 * numpy.concatenate( [5*xyd[...,0,_]*q**2 /R**2 , (t - xyd[...,1,_] + 5*xyd[...,1,_]*q**2/R**2) , -(s - d + 5*d*q**2/R**2)], axis=-1 ) \
       + a7 * [-1,-1,1] * xyd
   
         
    dudx = duAdx - duAdxprime + duBdx + xyz[...,2,_]*duCdx 
    dudy = duAdy - duAdyprime + duBdy + xyz[...,2,_]*duCdy 
    dudz = duAdz + duAdzprime + duBdz + xyz[...,2,_]*duCdz+ uC

    return  numpy.array( [ dudx.T, dudy.T, dudz.T ]).T
    

class DipSource( Source ):
  def __init__( self, depth, dip=90 ):
    self.xyz = numpy.array([ 0, 0, -depth ])
    rad = dip * numpy.pi / 180        
    self.cosd  = numpy.cos(rad)
    self.cos2d = numpy.cos(2*rad)
    self.sind  = numpy.sin(rad)
    self.sin2d = numpy.sin(2*rad)
    
  def displacement( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )
      
    xyd      = xyz * [1,1,-1] - self.xyz
    xydprime = xyz - self.xyz

    d = xyd[...,2,_]
    c = -self.xyz[2]*numpy.ones_like(d)

    alpha = .5 / (1 - poisson)

    R      = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    Rprime = numpy.linalg.norm( xydprime , axis=-1 )[...,_]
    q      = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    qprime = xydprime[...,1,_]*self.sind - xydprime[...,2,_]*self.cosd   
    p      = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    pprime = xydprime[...,1,_]*self.cosd + xydprime[...,2,_]*self.sind    
    s      = p*self.sind + q*self.cosd
    sprime = pprime*self.sind + qprime*self.cosd
    t      = p*self.cosd - q*self.sind
    tprime = pprime*self.cosd - qprime*self.sind
    
    
    a1      = 0.5 * ( 1 - alpha ) / R**3
    a1prime = 0.5 * ( 1 - alpha ) / Rprime**3
    a2      = 1.5 * ( alpha*q*p ) / R**5
    a2prime = 1.5 * ( alpha*qprime*pprime ) / Rprime**5
    a3      = ( 3.0*p*q ) / R**5
    a4      = ( 1-alpha ) * self.sind * self.cosd / alpha
    a5      = ( 1-alpha ) / R**3
    a6      = alpha*3*c / R**5
    
    I1 =  xyd[...,1,_] * ( ( 1. / (R * (R+xyd[...,2,_])**2) ) - xyd[...,0,_]**2*( (3.0*R+xyd[...,2,_]) / (R**3 * (R+xyd[...,2,_])**3) ) )
    I2 =  xyd[...,0,_] * ( ( 1. / (R*(R+xyd[...,2,_])**2) ) - xyd[...,1,_]**2*( (3.0*R+xyd[...,2,_])/(R**3 *(R+xyd[...,2,_])**3) ) )
    I3 =  xyd[...,0,_] / R**3 - I2
    I5 =  1. / ( R*(R+xyd[...,2,_]) ) - xyd[...,0,_]**2 * ( 2*R+xyd[...,2,_] ) / ( R**3 * (R+xyd[...,2,_])**2 )

    A3      = 1.0 - 3.0*xyd[...,0,_]**2 / R**2
     
    uA =  a1 * numpy.concatenate( [numpy.zeros_like(s), s, -t], axis=-1 ) \
       +  a2 * xyd
    uAprime =  a1prime * numpy.concatenate( [numpy.zeros_like(sprime), sprime, -tprime], axis=-1 ) \
            +  a2prime * xydprime            
    uB = -a3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) \
       +  a4 * numpy.concatenate( [I3, I1, I5], axis=-1 )  
        
    uC =  a5 * numpy.concatenate( [3*xyd[...,0,_]*t / R**2, \
                            -(self.cos2d - 3*xyd[...,1,_]*t / R**2), \
                            -A3*self.sind *self.cosd], axis=-1 ) \
       +  a6 * numpy.concatenate( [-5*xyd[...,0,_]*p*q/R**2, s-5*xyd[...,1,_]*p*q/R**2, t+5*d*p*q/R**2], axis=-1 )
    
    return uA - uAprime + uB + xyz[...,2,_]*uC
    
  def gradient( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )
    
    xyd      = xyz * [1,1,-1] - self.xyz
    xydprime = xyz - self.xyz
    

    alpha  = .5 / ( 1 - poisson )
    R      = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    Rprime = numpy.linalg.norm( xydprime, axis=-1 )[...,_]
    q      = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    qprime = xydprime[...,1,_]*self.sind - xydprime[...,2,_]*self.cosd
    p      = xyd[...,1,_]*self.cosd + xyd[...,2,_]*self.sind
    pprime = xydprime[...,1,_]*self.cosd + xydprime[...,2,_]*self.sind    
    s      = p*self.sind + q*self.cosd
    sprime = pprime*self.sind + qprime*self.cosd
    t      = p*self.cosd - q*self.sind
    tprime = pprime*self.cosd - qprime*self.sind

    A3      = 1.0 - 3.0*xyd[...,0,_]**2 / R**2   
    A5      = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    A5prime = 1.0 - 5.0*xydprime[...,0,_]**2 / Rprime**2
    A7      = 1.0 - 7.0*xyd[...,0,_]**2 / R**2
    
    B5 = 1-5. * xyd[...,1,_]**2 / R**2
    B7 = 1-7. * xyd[...,1,_]**2 / R**2
    
    C5 = 1-5. * xyd[...,2,_]**2 / R**2
    C7 = 1-7. * xyd[...,2,_]**2 / R**2
   
    d      = xyd[...,2,_]
    dprime = xydprime[...,2,_] 
    c      = -self.xyz[2]*numpy.ones_like(d)
    
    J1 =-3.0*xyd[...,0,_]*xyd[...,1,_] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0,_]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0,_]**2*xyd[...,1,_]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J3 = A3/R**3 - J2
    
    K1 =-xyd[...,1,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,0,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) ) 
    K2 =-xyd[...,0,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,1,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) )  
    K3 =-3.*xyd[...,0,_]*d/R**5 - K2
    
    V      = s - 5.*xyd[...,1,_]*p*q / R**2
    Vprime = sprime - 5.*xydprime[...,1,_]*pprime*qprime / Rprime**2
    Vpri   = t + 5.0*d*p*q/R**2
    Vpriprime = tprime + 5*dprime*pprime*qprime / Rprime**2
    
    a1      = 1.5 * (1 - alpha) * xyd[...,0,_] / R**5
    a1prime = 1.5 * (1 - alpha) * xydprime[...,0,_] / Rprime**5
    a2      = 1.5 * alpha*p*q / R**5
    a2prime = 1.5 * alpha*pprime*qprime / Rprime**5
    a3      = 3.*p*q / R**5
    a4      = (1-alpha)*self.sind*self.cosd / alpha
    a5      = (1-alpha)*3. / R**5
    a6      = alpha*15.*c / R**7
    
    b1      = 0.5 * (1-alpha) / R**3 
    b1prime = 0.5 * (1-alpha) / Rprime**3 
    b2      = 1.5 * alpha*V / R**5
    b2prime = 1.5 * alpha*Vprime / Rprime**5
    b3      = 3.*V / R**5
    b6      = alpha*3.*c / R**5
    
    c2      = 1.5*alpha*Vpri / R**5
    c2prime = 1.5*alpha*Vpriprime / Rprime**5    
    c3      = 3*Vpri / R**5
    
    d1      = ( 1-alpha ) / R**3
          
    
    duAdx = a1 * numpy.concatenate( [ numpy.zeros_like(s) , -s, t ], axis=-1 ) \
          + a2 * numpy.concatenate( [ A5 , -5*xyd[...,0,_]*xyd[...,1,_]/R**2 , -5*xyd[...,0,_]*d/R**2 ], axis=-1 )
    duAdxprime = a1prime * numpy.concatenate( [ numpy.zeros_like(sprime) , -sprime, tprime ], axis=-1 ) \
               + a2prime * numpy.concatenate( [ A5prime , -5*xydprime[...,0,_]*xydprime[...,1,_]/Rprime**2 , \
                                           -5*xydprime[...,0,_]*dprime/Rprime**2 ], axis=-1 )          
    duBdx = a3 * numpy.concatenate( [ -A5 , 5*xyd[...,0,_]*xyd[...,1,_]/R**2 , 5*c*xyd[...,0,_]/R**2 ], axis=-1 ) \
          + a4 * numpy.concatenate( [ J3, J1, K3], axis=-1 )       
          
    duCdx = a5 * numpy.concatenate( [ t*A5 , xyd[...,0,_] * (self.cos2d - 5*xyd[...,1,_]*t/R**2) , \
                               xyd[...,0,_] * (2+A5) * self.sind*self.cosd ], axis=-1 ) \
          - a6 * numpy.concatenate( [ p*q*A7 , xyd[...,0,_] * (s - 7*xyd[...,1,_]*p*q / R**2) , \
                                 xyd[...,0,_] * (t + 7*d*p*q/R**2)], axis=-1 )



    duAdy = b1 * numpy.concatenate( [ numpy.zeros_like(s) , self.sin2d - 3.*xyd[...,1,_]*s / R**2 , \
                               -(self.cos2d - 3.*xyd[...,1,_]*t / R**2) ], axis=-1) \
          + b2 * xyd \
          + a2 * [ 0, 1, 0] 
    duAdyprime = b1prime * numpy.concatenate( [ numpy.zeros_like(sprime) , self.sin2d - 3.*xydprime[...,1,_]*sprime / Rprime**2 , \
                                          -(self.cos2d - 3.*xydprime[...,1,_]*tprime / Rprime**2) ], axis=-1) \
               + b2prime * xydprime \
               + a2prime * [ 0, 1, 0] 
    duBdy =-b3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c], axis=-1 ) \
          - a3 * [ 0, 1, 0 ]  \
          + a4 * numpy.concatenate( [ J1, J2, K1 ], axis=-1 )    
    duCdy = a5 * numpy.concatenate( [ xyd[...,0,_] * (self.cos2d - 5*xyd[...,1,_]*t / R**2) , \
                                 2*xyd[...,1,_]*self.cos2d + t*B5 , \
                                 xyd[...,1,_]*A5*self.sind*self.cosd ], axis=-1 ) \
          + b6 * numpy.concatenate( [-(5*xyd[...,0,_]/R**2) * (s - 7.*xyd[...,1,_]*p*q / R**2) , \
                                 self.sin2d - 10.*xyd[...,1,_]*s / R**2 - 5.*p*q*B7 / R**2 , \
                                -((3+A5) * self.cos2d + 35.*xyd[...,1,_]*d*p*q / R**4) ], axis=-1 )




    duAdz = b1 * numpy.concatenate( [ numpy.zeros_like(s) , self.cos2d + 3*d*s/R**2 , self.sin2d - 3*d*t/R**2 ], axis=-1 ) \
          + c2 * xyd \
          - a2 * [ 0, 0, 1 ]
    duAdzprime = b1prime * numpy.concatenate( [ numpy.zeros_like(sprime) , self.cos2d + 3*dprime*sprime/Rprime**2 , 
                                           self.sin2d - 3*dprime*tprime/Rprime**2 ], axis=-1 ) \
               + c2prime * xydprime \
               - a2prime * [ 0, 0, 1 ]
    duBdz =-c3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) \
          + a4 * numpy.concatenate( [ -K3 , -K1 , A3/R**3 ], axis=-1 )
    duCdz =-a5 * numpy.concatenate( [ xyd[...,0,_] * (self.sin2d - 5*d*t/R**2) , \
                                 d*B5*self.cos2d + xyd[...,1,_]*C5*self.sin2d, \
                                 d*A5*self.sind*self.cosd ], axis=-1 ) \
          - b6 * numpy.concatenate( [ (5*xyd[...,0,_]/R**2) * (t + 7*d*p*q/R**2) , \
                                 (3+A5) * self.cos2d + 35.*xyd[...,1,_]*d*p*q/R**4 , \
                                 self.sin2d - 10*d*t/R**2 + 5*p*q*C7/R**2 ], axis=-1 )          
                                                                                                                                                                                                                                                            
    uC =  d1 * numpy.concatenate( [3*xyd[...,0,_]*t / R**2, \
                            -(self.cos2d - 3*xyd[...,1,_]*t / R**2), \
                            -A3*self.sind *self.cosd], axis=-1 ) \
       +  b6 * numpy.concatenate( [-5*xyd[...,0,_]*p*q/R**2 , s-5*xyd[...,1,_]*p*q/R**2 , t+5*d*p*q/R**2], axis=-1 )   
    
    dudx = duAdx - duAdxprime + duBdx + xyz[...,2,_]*duCdx 
    dudy = duAdy - duAdyprime + duBdy + xyz[...,2,_]*duCdy 
    dudz = duAdz + duAdzprime + duBdz + xyz[...,2,_]*duCdz + uC
  
    return  numpy.array( [ dudx.T, dudy.T, dudz.T ]).T


class StrikeSource( Source ):
  def __init__( self, depth, dip=90 ):
    self.xyz = numpy.array([ 0, 0, -depth ])
    rad = dip * numpy.pi / 180
    self.cosd = numpy.cos(rad)
    self.sind = numpy.sin(rad)

  def displacement( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )
    
    xyd      = xyz * [1,1,-1] - self.xyz
    xydprime = xyz - self.xyz

    d = xyd[...,2,_]
    c = -self.xyz[2]*numpy.ones_like(d)

    alpha = .5 / (1 - poisson)

    R      = numpy.linalg.norm( xyd , axis=-1 )[...,_]
    Rprime = numpy.linalg.norm( xydprime , axis=-1 )[...,_]
    q      = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    qprime = xydprime[...,1,_]*self.sind - xydprime[...,2,_]*self.cosd

    a1      = 0.5 * ( 1 - alpha ) / R**3
    a1prime = 0.5 * ( 1 - alpha ) / Rprime**3
    a2      = 1.5 * ( alpha*xyd[...,0,_]*q ) / R**5
    a2prime = 1.5 * ( alpha*xydprime[...,0,_]*qprime ) / Rprime**5
    a3 = ( 3.0*xyd[...,0,_]*q ) / R**5
    a4 = ( 1 - alpha )*self.sind / alpha
    a5 = ( 1 - alpha ) / R**5 
    a6 = alpha*3.0*c / R**5 

    I1 =  xyd[...,1,_] * ( ( 1. / (R * (R+xyd[...,2,_])**2) ) - xyd[...,0,_]**2*( (3.0*R+xyd[...,2,_]) / (R**3 * (R+xyd[...,2,_])**3) ) )
    I2 =  xyd[...,0,_] * ( ( 1. / (R*(R+xyd[...,2,_])**2) ) - xyd[...,1,_]**2*( (3.0*R+xyd[...,2,_])/(R**3 *(R+xyd[...,2,_])**3) ) )
    #I3 =  xyd[...,0,_] / R**3 - I2
    I4 = -xyd[...,0,_] * xyd[...,1,_] * ( 2*R+xyd[...,2,_] ) / ( R**3 * (R+xyd[...,2,_])**2 )
    #I5 =  1. / ( R*(R+xyd[...,2,_]) ) - xyd[...,0,_]**2 * ( 2*R+xyd[...,2,_] ) / ( R**3 * (R+xyd[...,2,_])**2 )
          
    uA      =  a1 * numpy.concatenate( [ q , xyd[...,0,_]*self.sind , -xyd[...,0,_]*self.cosd ], axis=-1 ) \
            +  a2 * xyd
    uAprime =  a1prime * numpy.concatenate( [ qprime , xydprime[...,0,_]*self.sind , -xydprime[...,0,_]*self.cosd ], axis=-1 ) \
            +  a2prime * xydprime
    uB = -a3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) \
       -  a4 * numpy.concatenate( [ I1, I2, I4 ], axis=-1 )   
    uC =  a5 * numpy.concatenate( [ -( R**2 - 3.0 * xyd[...,0,_]**2 ) * self.cosd ,      \
                                3.0 * xyd[...,0,_] * xyd[...,1,_] * self.cosd ,   \
                               -3.0 * xyd[...,0,_] * xyd[...,1,_] * self.sind ], axis=-1 ) \
       +  a6 * numpy.concatenate( [ q * ( 1 - (5 * xyd[...,0,_]**2 / R**2) ) , \
                               xyd[...,0,_] * ( self.sind - (5 * xyd[...,1,_] * q / R**2) ), \
                               xyd[...,0,_] * ( self.cosd + (5 * xyd[...,2,_] * q / R**2) ) ], axis=-1 )
   
    return uA - uAprime + uB + xyz[...,2,_]*uC 
    

  def gradient( self, xyz, poisson ):
    xyz = numpy.asarray( xyz )

    xyd      = xyz * [1,1,-1] - self.xyz
    xydprime = xyz - self.xyz

    alpha  = .5 / ( 1 - poisson )
    R      = numpy.linalg.norm( xyd, axis=-1 )[...,_]
    Rprime = numpy.linalg.norm( xydprime, axis=-1 )[...,_]
    q      = xyd[...,1,_]*self.sind - xyd[...,2,_]*self.cosd
    qprime = xydprime[...,1,_]*self.sind - xydprime[...,2,_]*self.cosd
    
    A3      = 1.0 - 3.0*xyd[...,0,_]**2 / R**2   
    A3prime = 1.0 - 3.0*xydprime[...,0,_]**2 / Rprime**2   
    A5      = 1.0 - 5.0*xyd[...,0,_]**2 / R**2
    A5prime = 1.0 - 5.0*xydprime[...,0,_]**2 / Rprime**2
    A7      = 1.0 - 7.0*xyd[...,0,_]**2 / R**2
    
    B5 = 1-5. * xyd[...,1,_]**2 / R**2
    B7 = 1-7. * xyd[...,1,_]**2 / R**2
    
    C7 = 1-7. * xyd[...,2,_]**2 / R**2
   
    d = xyd[...,2,_]
    c = -self.xyz[2]*numpy.ones_like(d)
    
    J1 =-3.0*xyd[...,0,_]*xyd[...,1,_] * ( (3.0*R + d) / (R**3 * (R+d)**3) - xyd[...,0,_]**2 * (5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4) )
    J2 = (1./R**3) - 3.0/(R * (R+d)**2) + 3.0*xyd[...,0,_]**2*xyd[...,1,_]**2*(5*R**2 + 4*R*d + d**2) / (R**5 * (R+d)**4)
    J4 =-3.0*xyd[...,0,_]*xyd[...,1,_] / R**5 - J1
    
    K1 =-xyd[...,1,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,0,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) ) 
    K2 =-xyd[...,0,_] * ( (2. * R + d) / (R**3 * (R + d)**2) - xyd[...,1,_]**2 * (8.0 * R**2 + 9. * R * d + 3.0 * d**2) / (R**5 * (R + d)**3) )  

    U         = self.sind - 5*xyd[...,1,_]*q / R**2
    Uprime    = self.sind - 5*xydprime[...,1,_]*qprime / Rprime**2
    Upri      = self.cosd + 5*d*q / R**2
    Upriprime = self.cosd + 5*xydprime[...,2,_]*qprime / Rprime**2
    
    a1      = 0.5 * (1 - alpha) / R**3
    a1prime = 0.5 * (1 - alpha) / Rprime**3
    a2      = 1.5 * (alpha * q) / R**5
    a2prime = 1.5 * (alpha * qprime) / Rprime**5
    
    a3 = (3.0 * q) / R**5
    a4 = (1 - alpha) * self.sind / alpha
    a5 = 3 * (1 - alpha) / R**5 
    a6 = alpha*3.0*c / R**5 
    
    b2      = 1.5* alpha*xyd[...,0,_]*U / R**5
    b2prime = 1.5* alpha*xydprime[...,0,_]*Uprime / Rprime**5
    b3      = 3*xyd[...,0,_]*U / R**5
    
    c2      = 1.5*alpha*xyd[...,0,_]*Upri / R**5
    c2prime = 1.5* alpha*xydprime[...,0,_]*Upriprime / Rprime**5
    c3      = 3*xyd[...,0,_]*Upri / R**5
          
    duAdx      = a1 * numpy.concatenate( [ -3.0*xyd[...,0,_]*q / R**2 , A3*self.sind , -A3*self.cosd ], axis=-1 ) \
               + a2 * xyd * ( A5 + [1,0,0] )
    duAdxprime = a1prime * numpy.concatenate( [ -3.0*xydprime[...,0,_]*qprime / Rprime**2 , A3prime*self.sind , -A3prime*self.cosd ], axis=-1 ) \
               + a2prime * xydprime * ( A5prime + [1,0,0] )
          
    duBdx =-a3 * ( A5 + [1,0,0] ) * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) \
          - a4 * numpy.concatenate( [ J1, J2, K1 ], axis=-1 )
    duCdx = a5 * ( A5 + [2,0,0] ) * numpy.concatenate( [ xyd[...,0,_]*self.cosd , xyd[...,1,_]*self.cosd , -xyd[...,1,_]*self.sind ], axis=-1 ) \
          + a6 * numpy.concatenate( [ -5*xyd[...,0,_]*q*(2+A7) / R**2 , A5*self.sind - 5*xyd[...,1,_]*q*A7 / R**2 , A5*self.cosd + 5*d*q*A7 / R**2 ], axis=-1 )
    duAdy = a1 * numpy.concatenate( [ self.sind - 3.0*xyd[...,1,_]*q / R**2 , -3.0*xyd[...,0,_]*xyd[...,1,_]*self.sind / R**2 , 3.0*xyd[...,0,_]*xyd[...,1,_]*self.cosd / R**2 ], axis=-1 ) \
          + b2 * xyd \
          + a2 * [0,1,0] * xyd[...,0,_]
    duAdyprime = a1prime * numpy.concatenate( [ self.sind - 3.0*xydprime[...,1,_]*qprime / Rprime**2 ,      \
                                         -3.0*xydprime[...,0,_]*xydprime[...,1,_]*self.sind / Rprime**2,    \
                                          3.0*xydprime[...,0,_]*xydprime[...,1,_]*self.cosd / Rprime**2 ], axis=-1 ) \
               + b2prime * xydprime \
               + a2prime * [0,1,0] * xydprime[...,0,_]
    duBdy =-b3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) \
          - a3 * [0,1,0]* xyd[...,0,_]\
          - a4 * numpy.concatenate( [ J2, J4, K2 ], axis=-1 )       
    duCdy = a5 * numpy.concatenate( [ xyd[...,1,_]*A5*self.cosd , xyd[...,0,_]*B5*self.cosd , -xyd[...,0,_]*B5*self.sind ], axis=-1 ) \
          + a6 * numpy.concatenate( [ A5*self.sind - 5*xyd[...,1,_]*q*A7 / R**2 ,        \
                               -5*xyd[...,0,_]*(2*xyd[...,1,_]*self.sind + q*B7) / R**2, \
                                5*xyd[...,0,_]*(d*B7*self.sind - xyd[...,1,_]*C7*self.cosd) / R**2 ], axis=-1 )

    duAdz = a1 * numpy.concatenate( [ self.cosd + 3*d*q / R**2 , 3*xyd[...,0,_]*d*self.sind / R**2 , -3*xyd[...,0,_]*d*self.cosd / R**2 ], axis=-1 )\
          + c2 * xyd \
          - a2 * [0,0,1] * xyd[...,0,_]
    duAdzprime = a1prime * numpy.concatenate( [ self.cosd +3*xydprime[...,2,_]*qprime / Rprime**2 ,         \
                                          3*xydprime[...,0,_]*xydprime[...,2,_]*self.sind / Rprime**2,  \
                                         -3*xydprime[...,0,_]*xydprime[...,2,_]*self.cosd / Rprime**2 ], axis=-1 )\
               + c2prime * xydprime\
               - a2prime * [0,0,1] * xydprime[...,0,_]
    
    duBdz =-c3 * numpy.concatenate( [ xyd[...,0,_] , xyd[...,1,_] , c ], axis=-1 ) \
          + a4 * numpy.concatenate( [ K1 , K2 , 3*xyd[...,0,_]*xyd[...,1,_] / R**5 ], axis=-1 ) 
    duCdz = a5 * numpy.concatenate( [ -d*A5*self.cosd , d*5*xyd[...,0,_]*xyd[...,1,_]*self.cosd / R**2 , -d*5*xyd[...,0,_]*xyd[...,1,_]*self.sind / R**2 ], axis=-1 ) \
          + a6 * numpy.concatenate( [ A5*self.cosd+5*d*q*A7 / R**2 ,                          \
                                5*xyd[...,0,_]*(d*B7*self.sind - xyd[...,1,_]*C7*self.cosd) / R**2, \
                                5*xyd[...,0,_]*(2*d*self.cosd - q*C7) / R**2 ], axis=-1 )                 
                                                      
    uC =  a5 * numpy.concatenate( [-( R**2/3.0 - xyd[...,0,_]**2 )*self.cosd , xyd[...,0,_]*xyd[...,1,_]*self.cosd , -xyd[...,0,_]*xyd[...,1,_]*self.sind ], axis=-1 ) \
       +  a6 * numpy.concatenate( [ q*( 1 - (5*xyd[...,0,_]**2 / R**2) ) , xyd[...,0,_]*( self.sind - (5*xyd[...,1,_]*q / R**2) ) , xyd[...,0,_]*(self.cosd + (5*xyd[...,2,_]*q / R**2)) ], axis=-1 )   
    
    dudx = duAdx - duAdxprime + duBdx + xyz[...,2,_]*duCdx 
    dudy = duAdy - duAdyprime + duBdy + xyz[...,2,_]*duCdy 
    dudz = duAdz + duAdzprime + duBdz + xyz[...,2,_]*duCdz + uC
  
    return  numpy.array( [ dudx.T, dudy.T, dudz.T ]).T
