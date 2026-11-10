#!/usr/bin/env python3
"""Check the native workspace has one formatter and one type checker."""
import re
import subprocess
from pathlib import Path
import tomllib
import yaml

REPO = Path(__file__).resolve().parents[2]


def main():
    config = yaml.safe_load((REPO/'.pre-commit-config.yaml').read_text())
    hooks = {hook['id']: hook for repository in config['repos'] for hook in repository['hooks']}
    assert not ({'black','isort','pylint','mypy'} & hooks.keys())
    for name in ('ruff-check','ruff-format','ty'):
        assert hooks[name].get('files'), f'{name} has no explicit scope'
    files = subprocess.check_output(['git','ls-files','*.py'],cwd=REPO,text=True).splitlines()
    files = [name for name in files if (REPO/name).exists()]
    for name in files:
        if name.startswith(('packages/','tests/parity/','tests/retirement/')) or name=='tests/conftest.py':
            assert re.match(hooks['ruff-check']['files'],name), name
            assert re.match(hooks['ruff-format']['files'],name), name
        assert bool(re.match(hooks['ty']['files'],name)) == name.startswith('packages/'), name
    dev = tomllib.loads((REPO/'pyproject.toml').read_text())['dependency-groups']['dev']
    ci = (REPO/'.github/workflows/ci-core.yaml').read_text()
    for tool in ('ruff','ty'):
        pin = next(item for item in dev if isinstance(item,str) and item.startswith(tool+'=='))
        assert pin in ci, f'CI disagrees with {pin}'
        if tool=='ruff':
            version=next(repo['rev'].removeprefix('v') for repo in config['repos'] if 'ruff-pre-commit' in repo['repo'])
            assert pin == 'ruff=='+version
    print(f'Native toolchain ownership verified for {len(files)} Python files')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
