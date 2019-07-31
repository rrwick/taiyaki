import numpy as np
import os
from subprocess import Popen, PIPE

# Data and script paths relative to working directory build/acctest
BIN_DIR = "../../bin"
DATA_DIR = "../../test/data"
MISC_DIR = "../../misc"
MODELS_DIR = "../../models"

class Result(object):

    def __init__(self, test_case, cmd, cwd, exit_code, stdout, stderr, max_lines=100):
        self.test_case = test_case
        self.cmd = cmd
        self.cwd = cwd
        self._exit_code = exit_code
        self._stdout = stdout.strip('\n').split('\n')
        self._stderr = stderr.strip('\n').split('\n')
        self.max_lines = max_lines

    def __repr__(self):
        L = ['\n\tCommand: {}'.format(' '.join(self.cmd))]
        if self.cwd:
            L.append('\n\tCwd: {}'.format(self.cwd))

        if self._exit_code:
            L.append('\tCommand exit code: %s' % self._exit_code)

        if self._stdout:
            L.append('\n\tFirst {} lines of stdout:'.format(self.max_lines))
            for line in self._stdout[:self.max_lines]:
                L.append("\t\t{}".format(line))

        if self._stderr:
            L.append('\n\tFirst {} lines of stderr:'.format(self.max_lines))
            for line in self._stderr[:self.max_lines]:
                L.append("\t\t{}".format(line))

        return '\n'.join(L)

    def expect_exit_code(self, expected_exit_code):
        msg = "expected return code %s but got %s in: %s" % (expected_exit_code, self._exit_code, self)
        self.test_case.assertEqual(expected_exit_code, self._exit_code, msg)
        return self

    def expect_stdout(self, f):
        msg = "expectation on stdout failed for: %s" % self
        self.test_case.assertTrue(f(self._stdout), msg)
        return self

    def expect_stdout_equals(self, referenceStdout):
        self.test_case.assertEquals(self._stdout, referenceStdout)
        return self

    def expect_stderr(self, f):
        msg = "expectation on stderr failed for: %s" % self
        self.test_case.assertTrue(f(self._stderr), msg)
        return self

    def expect_stderr_equals(self, referenceStderr):
        self.test_case.assertEquals(self._stderr, referenceStderr)
        return self

    def get_exit_code(self):
        return self._exit_code

    def get_stdout(self):
        return self._stdout

    def get_stderr(self):
        return self._stderr


def run_cmd(test_case, cmd, cwd=None):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
    stdout, stderr = proc.communicate(None)

    exit_code = proc.returncode
    stdout = stdout.decode('UTF-8')
    stderr = stderr.decode('UTF-8')

    return Result(test_case, cmd, cwd, exit_code, stdout, stderr)


def maybe_create_dir(directory_name):
    '''
    Create a directory if it does not exist already.
    In Python 2.7 OSError is thrown if directory does not exist or permissions are insufficient.
    In Python 3 more specific exceptions are thrown.
    '''

    try:
        os.makedirs(directory_name)
    except OSError:
        if os.path.exists(directory_name) and os.path.isdir(directory_name):
            pass
        else:
            raise


def any_line_starts_with(prefix):
    return lambda lines: any(l.startswith(prefix) for l in lines)


def assertArrayEqual(test_case, a, b):
    test_case.assertEqual(a.shape, b.shape,
                          msg='Array shape mismatch: {} != {}\na = {}\nb = {}'.format(a.shape, b.shape, a, b))
    test_case.assertTrue(np.array_equal(a, b),
                         msg='Array element mismatch: {} != {}\nshape = {}'.format(a, b, a.shape))


if __name__ == '__main__':
    assert not zeroth_line_starts_with('a')([])
    assert zeroth_line_starts_with('a')(['a'])
    assert zeroth_line_starts_with('a')(['a', 'a'])
    assert zeroth_line_starts_with('a')(['a', 'b'])
    assert not zeroth_line_starts_with('a')(['b', 'a'])

    assert not last_line_starts_with('a')([])
    assert last_line_starts_with('a')(['a'])
    assert last_line_starts_with('a')(['a', 'a'])
    assert not last_line_starts_with('a')(['a', 'b'])
    assert last_line_starts_with('a')(['b', 'a'])
