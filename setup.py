#! /usr/bin/env python3

import setuptools, setuptools.command.build_py
import os, platform

compilecmd = {
  'Linux': 'gcc -shared -o {dst:}.so -lc -fpic -g -O3 -march=native -mtune=native {src:}.c',
  'Windows': 'cl /LD /DDLL {src:}.c /OUT:{dst:}.dll',
}[ platform.system() ]

class build_py( setuptools.command.build_py.build_py ):
  def run(self):
    setuptools.command.build_py.build_py.run( self )
    print( 'compiling libokada' )
    assert 0 == os.system( compilecmd.format(
      src=os.path.join('halfspace','libokada'),
      dst=os.path.join(self.build_lib,'halfspace','libokada') ) )

setuptools.setup(
  name='halfspace',
  version='1.9',
  author='Gertjan van Zwieten',
  packages=[ 'halfspace' ],
  cmdclass={ 'build_py': build_py },
)
