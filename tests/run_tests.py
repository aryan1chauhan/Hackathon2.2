import importlib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tests.test_env as test_env


def run_all():
    failures = []
    for name in dir(test_env):
        if name.startswith('test_') and callable(getattr(test_env, name)):
            try:
                print('Running', name)
                getattr(test_env, name)()
                print('  OK')
            except AssertionError as e:
                print('  FAIL:', e)
                failures.append((name, e))
            except Exception as e:
                print('  ERROR:', e)
                failures.append((name, e))
    if failures:
        print('\nSome tests failed:')
        for name, err in failures:
            print(' -', name, err)
        sys.exit(1)
    print('\nAll tests passed')


if __name__ == '__main__':
    run_all()
