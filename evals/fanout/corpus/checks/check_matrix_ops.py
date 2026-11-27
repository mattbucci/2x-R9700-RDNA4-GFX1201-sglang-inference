"""Behaviour checks for matrix_ops. Usage: check_matrix_ops.py <module path>."""
import importlib.util
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raises(exc, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc:
        return True
    return False



def main():
    import itertools
    import random
    from fractions import Fraction

    m = load(sys.argv[1], "matrix_ops_under_test")
    Matrix, MatrixError = m.Matrix, m.MatrixError

    a = Matrix([[1, 2, 3], [4, 5, 6]])
    assert a.shape == (2, 3) and not a.is_square
    assert a.transpose() == Matrix([[1, 4], [2, 5], [3, 6]])
    assert a.transpose().transpose() == a
    assert a.row(1) == [4, 5, 6] and a.column(2) == [3, 6]
    assert a[1, 0] == 4
    assert raises(MatrixError, Matrix, [[1, 2], [3]])
    assert raises(MatrixError, Matrix, [])

    b = Matrix([[7, 8], [9, 10], [11, 12]])
    assert a.multiply(b) == Matrix([[58, 64], [139, 154]])
    assert (a @ b) == a.multiply(b)
    assert raises(MatrixError, a.multiply, a)
    assert a.add(a) == a.scale(2)
    assert a.subtract(a) == Matrix.zeros(2, 3)

    eye = Matrix.identity(3)
    assert eye.to_lists() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert eye.determinant() == 1 and eye.trace() == 3

    assert Matrix([[5]]).determinant() == 5
    assert Matrix([[1, 2], [3, 4]]).determinant() == -2
    assert Matrix([[2, 0, 1], [1, 3, 2], [1, 1, 2]]).determinant() == 6
    assert Matrix([[2, 0, 1], [1, 3, 2], [1, 1, 1]]).determinant() == 0
    assert Matrix([[1, 2], [2, 4]]).determinant() == 0
    assert Matrix([[0, 1], [1, 0]]).determinant() == -1
    assert Matrix([[0, 1], [-2, 3]]).determinant() == 2
    assert Matrix([[-1, 2, 0], [-3, 1, 4], [0, 5, -2]]).determinant() == 10
    assert Matrix([[0, 0, 2], [0, -3, 1], [-4, 1, 1]]).determinant() == -24
    assert Matrix([[Fraction(1, 2), 1], [1, 4]]).determinant() == 1
    assert abs(Matrix([[1.5, 2.0], [0.5, 4.0]]).determinant() - 5.0) < 1e-12
    assert raises(MatrixError, a.determinant)

    def leibniz(rows):
        total = 0
        for perm in itertools.permutations(range(len(rows))):
            inversions = sum(
                1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j]
            )
            term = -1 if inversions % 2 else 1
            for r, c in enumerate(perm):
                term *= rows[r][c]
            total += term
        return total

    rng = random.Random(20240311)
    for _ in range(200):
        size = rng.randint(1, 5)
        rows = [[rng.randint(-4, 4) for _ in range(size)] for _ in range(size)]
        assert Matrix(rows).determinant() == leibniz(rows), rows

    f = Matrix([[1, 1], [1, 0]])
    assert f.power(0) == Matrix.identity(2)
    assert f.power(1) == f
    assert f.power(10) == Matrix([[89, 55], [55, 34]])
    assert Matrix([[2, 0], [0, 3]]).power(5) == Matrix([[32, 0], [0, 243]])
    assert raises(MatrixError, f.power, -1)
    assert raises(MatrixError, a.power, 2)
    print("ok")


if __name__ == "__main__":
    main()
