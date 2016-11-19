def mogi( xyz ):
  assert len(xyz) == 3
  from .nucleus import MogiSource
  return MogiSource( depth=-xyz[2] ).translated( xyz[:2] )

def couple( xyz, strength, dip=90, strike=0 ):
  assert len(strength) == 3 and any(strength)
  assert len(xyz) == 3
  from .nucleus import StrikeSource, DipSource, TensileSource
  sources = [ Source( depth=-xyz[2], dip=dip ) * s for Source, s in zip( [ StrikeSource, DipSource, TensileSource ], strength ) if s ]
  source = sources[0]
  for other in sources[1:]:
    source += other
  return source.rotated( strike ).translated( xyz[:2] )

def okada( **kwargs ):
  from .planar import OkadaSource
  return OkadaSource( **kwargs )
