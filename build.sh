#!/bin/bash
# Build & Install Package (& Test if desired)

# python3 -m build
# pip install dist/corescpy-0.2.0.tar.gz
# python3 setup.py install
# python3 setup.py develop

# if [[ $1 = 'test' ]]
#     then
#         py.test
# fi

pip install .
