def mogi( xyz ):
  assert len(xyz) == 3
  from .nucleus import MogiSource
  return MogiSource( depth=-xyz[2] ).translated( xyz[:2] )

def okada( **kwargs ):
  from .planar import OkadaSource
  return OkadaSource( **kwargs )
