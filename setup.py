# Following Nick Hand's https://github.com/bccp/nbodykit
from distutils.core import setup
from distutils.util import convert_path
import os

NAME='legacyzpts'

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

def find_packages(base_path):
    base_path = convert_path(base_path)
    found = []
    for root, dirs, files in os.walk(base_path, followlinks=True):
        dirs[:] = [d for d in dirs if d[0] != '.' and d not in ('ez_setup', '__pycache__')]
        relpath = os.path.relpath(root, base_path)
        parent = relpath.replace(os.sep, '.').lstrip('.')
        if relpath != '.' and parent not in found:
            # foo.bar package but no foo package, skip
            continue
        for dir in dirs:
            if os.path.isfile(os.path.join(root, dir, '__init__.py')):
                package = '.'.join((parent, dir)) if parent else dir
                found.append(package)
    return found

# the base dependencies
with open('requirements.txt', 'r') as fh:
    dependencies = [l.strip() for l in fh]

# extra dependencies
extras = {}
with open('requirements-extras.txt', 'r') as fh:
    extras['extras'] = [l.strip() for l in fh][1:]
    extras['full'] = extras['extras'] #

setup(name=NAME,
      version=find_version("py/legacyzpts/version.py"),
      author="Kaylan Burleigh, John Mustakas, et al",
      maintainer="Kaylan Burleigh",
      maintainer_email="kburleigh@lbl.gov",
      description="blah",
      url="http://github.com/legacysurvey/legacyzpts",
      zip_safe=False,
      packages = find_packages('py'),
      package_dir = {'':'py'},
      license='BSD',
      install_requires=dependencies,
      extras_require=extras
)
