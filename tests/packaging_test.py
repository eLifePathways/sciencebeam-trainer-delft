"""Checks that what the package imports is what the package declares.

Every one of these arrived transitively at some point, through delft or
scikit-learn or TensorFlow, and a resolution change is enough to remove one.
"""
import ast
import pathlib
import sys
import tomllib
from typing import Dict, Set

import pytest


PACKAGE_DIRECTORY = pathlib.Path(__file__).parent.parent / 'sciencebeam_trainer_delft'
PYPROJECT_PATH = pathlib.Path(__file__).parent.parent / 'pyproject.toml'

# import name -> distribution name, where they differ
DISTRIBUTION_NAME_BY_IMPORT_NAME = {
    'sklearn': 'scikit-learn',
    'typing_extensions': 'typing-extensions'
}

LOCAL_MODULE_NAMES = {'sciencebeam_trainer_delft', 'tests'}


def _iter_imported_names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                yield node.module.split('.')[0]


def get_imported_distribution_names() -> Dict[str, str]:
    """Maps each third-party distribution imported to a file that imports it."""
    imported: Dict[str, str] = {}
    for path in sorted(PACKAGE_DIRECTORY.rglob('*.py')):
        for name in _iter_imported_names(ast.parse(path.read_text())):
            if name in sys.stdlib_module_names or name in LOCAL_MODULE_NAMES:
                continue
            distribution_name = DISTRIBUTION_NAME_BY_IMPORT_NAME.get(name, name)
            imported.setdefault(distribution_name, str(path))
    return imported


def get_declared_distribution_names() -> Set[str]:
    with open(PYPROJECT_PATH, 'rb') as fp:
        pyproject = tomllib.load(fp)
    project = pyproject['project']
    requirements = list(project['dependencies'])
    for extra_requirements in project.get('optional-dependencies', {}).values():
        requirements.extend(extra_requirements)
    # torch is declared in a dependency group rather than an extra, since the
    # wheel variant cannot be selected through published metadata
    for group_requirements in pyproject.get('dependency-groups', {}).values():
        requirements.extend(group_requirements)
    return {
        requirement.split(';')[0].split('>')[0].split('=')[0].split('[')[0].strip()
        for requirement in requirements
    }


class TestDeclaredDependencies:
    def test_should_declare_every_imported_distribution(self):
        declared = get_declared_distribution_names()
        undeclared = {
            name: path
            for name, path in get_imported_distribution_names().items()
            if name not in declared
        }
        assert not undeclared

    @pytest.mark.parametrize('name', ['joblib', 'numpy', 'scikit-learn', 'fsspec'])
    def test_should_declare_the_dependencies_that_were_once_transitive(self, name: str):
        assert name in get_declared_distribution_names()

    def test_should_find_the_imports_it_is_checking(self):
        # the check is vacuous if the walk finds nothing
        assert len(get_imported_distribution_names()) >= 8
