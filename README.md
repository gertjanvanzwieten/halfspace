Halfspace
=========

The halfspace tool can be used to compute the displacements, strains, and
stresses, induced by any number of sources in a homogeneous elastic halfspace.
Currently the only supported source is the Okada rectangular dislocation plane
as originally published [here] [okada] (paywall). A mogi / inflation source
will be added in due time.

[okada]: http://www.bssaonline.org/content/82/2/1018.short


OkadaSource
-----------

The OkadaSource object requires:

  * length: length of the fault in strike direction [m]

And any seven of the following parameters (where xtrace-ytrace,
xtop-ytop, xbottom-ybottom and strike-dip must occur in pairs):

  * width: width of the fault in dip direction [m]
  * strike: strike angle [deg]
  * dip: dip angle [deg]
  * xtrace: x coordinate of the surface trace center [m]
  * ytrace: x coordinate of the surface trace center [m]
  * xtop: x coordinate of the top center [m]
  * ytop: y coordinate of the top center [m]
  * ztop: z coordinate of the top center [m]
  * xbottom: x coordinate of the bottom center [m]
  * ybottom: x coordinate of the bottom center [m]
  * zbottom: x coordinate of the bottom center [m]

And any two or three of the following (where slip-rake and
strikeslip-dipslip must occur in pairs, and opening defaults to zero):

  * slip: slip magnitude [m]
  * rake: rake angle [deg]
  * strikeslip: displacement jump in strike direction [m]
  * dipslip: displacement jump in dip direction [m]
  * opening: displacement jump in normal direction [m]

For example:

    >> okada = halfspace.OkadaSource( length=16e3, strike=30, dip=60,
                   ztop=-1.5e3, zbottom=-7e3, xtrace=0, ytrace=0,
                        slip=1.5, rake=0 )

All parameters can be retrieved from the object as okada.length, okada.strike,
etc.


Source functionality
--------------------

Source objects share the following methods:

  * displacement: 3D displacement vector
  * gradient: 3x3 displacement gradient
  * strain: 3x3 symmetric strain tensor
  * stress: 3x3 symmetric stress tensor

All take a set of coordinates as their first arguments, of arbitrary shape,
except for the last axis that should be of length 3 for x, y and z. The return
shape will be identical for displacements, or will have one axis appended for
the additional tensor dimension.

Second argument is the poisson ratio. For displacement, gradient, and strain,
this is the only determining parameter. For stress additionally the Young's
modulus of elasticity should be specified.

    >> u = okada.displacement( [1e3,1e4,0] ) # u.shape = 3,
    >> G = okada.gradient( [[1e3,2e3,0],[0,1e3,0]] ) # G.shape = 2,3,3

Furthermore, sources can be added and scaled using the + and * operators.

    >> multi = okada + mogi * 10

The resulting MultiSource object has the same functionalities as the
individial source objects, and can be used to quickly evaluate any
linear combination. The individual sources can be retrieved using the
[] operator

    >> multi[0] is okada # True

For further usage examples please see the scripts directory.
