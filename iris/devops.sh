#!/bin/sh

# Import the environment variables.
. ./.env

# Run all the hooks.
pre-commit run

# Run tests.
pytest

# Clean up artifacts
./clean.sh

# Generate the distribution and install the package locally [OPTIONAL].
for arg in "$@"
do
    if [ "$arg" = "pip" ] ; then
	python3 -m pip install --upgrade build
	pip3 uninstall -y $IMAGE_NAME
	python3 -m build
	python3 -m pip install --user .	
    fi
done


