package palladium.blas

import scala.scalanative.unsafe.*

/** Tests for the raw CBLAS FFI bindings.
  * Verifies that we can call into the system BLAS library correctly.
  */
class CblasSuite extends munit.FunSuite:

  val eps = 1e-10

  test("dnrm2: L2 norm of [1, 2, -2] = 3.0") {
    Zone.acquire { implicit z =>
      val x = alloc[Double](3)
      x(0) = 1.0; x(1) = 2.0; x(2) = -2.0
      val result = Cblas.cblas_dnrm2(3, x, 1)
      assertEqualsDouble(result, 3.0, eps)
    }
  }

  test("ddot: dot product of [1,2,3] . [4,5,6] = 32") {
    Zone.acquire { implicit z =>
      val x = alloc[Double](3)
      val y = alloc[Double](3)
      x(0) = 1.0; x(1) = 2.0; x(2) = 3.0
      y(0) = 4.0; y(1) = 5.0; y(2) = 6.0
      val result = Cblas.cblas_ddot(3, x, 1, y, 1)
      assertEqualsDouble(result, 32.0, eps)
    }
  }

  test("dscal: scale [1, 2, 3] by 2.0") {
    Zone.acquire { implicit z =>
      val x = alloc[Double](3)
      x(0) = 1.0; x(1) = 2.0; x(2) = 3.0
      Cblas.cblas_dscal(3, 2.0, x, 1)
      assertEqualsDouble(x(0), 2.0, eps)
      assertEqualsDouble(x(1), 4.0, eps)
      assertEqualsDouble(x(2), 6.0, eps)
    }
  }

  test("dcopy: copy vector") {
    Zone.acquire { implicit z =>
      val x = alloc[Double](3)
      val y = alloc[Double](3)
      x(0) = 1.0; x(1) = 2.0; x(2) = 3.0
      Cblas.cblas_dcopy(3, x, 1, y, 1)
      assertEqualsDouble(y(0), 1.0, eps)
      assertEqualsDouble(y(1), 2.0, eps)
      assertEqualsDouble(y(2), 3.0, eps)
    }
  }

  test("daxpy: y = 2.0 * x + y") {
    Zone.acquire { implicit z =>
      val x = alloc[Double](3)
      val y = alloc[Double](3)
      x(0) = 1.0; x(1) = 2.0; x(2) = 3.0
      y(0) = 10.0; y(1) = 20.0; y(2) = 30.0
      Cblas.cblas_daxpy(3, 2.0, x, 1, y, 1)
      assertEqualsDouble(y(0), 12.0, eps)
      assertEqualsDouble(y(1), 24.0, eps)
      assertEqualsDouble(y(2), 36.0, eps)
    }
  }

  test("dgemv: matrix-vector multiply A * x") {
    Zone.acquire { implicit z =>
      // A = [[1, 2], [3, 4], [5, 6]]  (3x2, row-major)
      // x = [1, 2]
      // y = A * x = [5, 11, 17]
      val a = alloc[Double](6)
      a(0) = 1.0; a(1) = 2.0
      a(2) = 3.0; a(3) = 4.0
      a(4) = 5.0; a(5) = 6.0
      val x = alloc[Double](2)
      x(0) = 1.0; x(1) = 2.0
      val y = alloc[Double](3)
      y(0) = 0.0; y(1) = 0.0; y(2) = 0.0

      Cblas.cblas_dgemv(
        CblasConstants.CblasRowMajor, CblasConstants.CblasNoTrans,
        3, 2,       // M=3 rows, N=2 cols
        1.0,        // alpha
        a, 2,       // A, lda=2
        x, 1,       // x, incX=1
        0.0,        // beta
        y, 1        // y, incY=1
      )

      assertEqualsDouble(y(0), 5.0, eps)
      assertEqualsDouble(y(1), 11.0, eps)
      assertEqualsDouble(y(2), 17.0, eps)
    }
  }

  test("dgemm: matrix-matrix multiply C = A * B") {
    Zone.acquire { implicit z =>
      // A = [[1, 2], [3, 4]]  (2x2)
      // B = [[5, 6], [7, 8]]  (2x2)
      // C = A * B = [[19, 22], [43, 50]]
      val a = alloc[Double](4)
      a(0) = 1.0; a(1) = 2.0; a(2) = 3.0; a(3) = 4.0
      val b = alloc[Double](4)
      b(0) = 5.0; b(1) = 6.0; b(2) = 7.0; b(3) = 8.0
      val c = alloc[Double](4)
      c(0) = 0.0; c(1) = 0.0; c(2) = 0.0; c(3) = 0.0

      Cblas.cblas_dgemm(
        CblasConstants.CblasRowMajor,
        CblasConstants.CblasNoTrans, CblasConstants.CblasNoTrans,
        2, 2, 2,    // M=2, N=2, K=2
        1.0,        // alpha
        a, 2,       // A, lda=2
        b, 2,       // B, ldb=2
        0.0,        // beta
        c, 2        // C, ldc=2
      )

      assertEqualsDouble(c(0), 19.0, eps)
      assertEqualsDouble(c(1), 22.0, eps)
      assertEqualsDouble(c(2), 43.0, eps)
      assertEqualsDouble(c(3), 50.0, eps)
    }
  }

  test("dgemm: non-square matrix multiply (3x2) * (2x4) = (3x4)") {
    Zone.acquire { implicit z =>
      // A = [[1, 2], [3, 4], [5, 6]]  (3x2)
      // B = [[1, 0, 1, 0], [0, 1, 0, 1]]  (2x4)
      // C = A * B = [[1, 2, 1, 2], [3, 4, 3, 4], [5, 6, 5, 6]]
      val a = alloc[Double](6)
      a(0) = 1.0; a(1) = 2.0
      a(2) = 3.0; a(3) = 4.0
      a(4) = 5.0; a(5) = 6.0
      val b = alloc[Double](8)
      b(0) = 1.0; b(1) = 0.0; b(2) = 1.0; b(3) = 0.0
      b(4) = 0.0; b(5) = 1.0; b(6) = 0.0; b(7) = 1.0
      val c = alloc[Double](12)
      var i = 0; while i < 12 do { c(i) = 0.0; i += 1 }

      Cblas.cblas_dgemm(
        CblasConstants.CblasRowMajor,
        CblasConstants.CblasNoTrans, CblasConstants.CblasNoTrans,
        3, 4, 2,    // M=3, N=4, K=2
        1.0, a, 2, b, 4, 0.0, c, 4
      )

      assertEqualsDouble(c(0), 1.0, eps)
      assertEqualsDouble(c(1), 2.0, eps)
      assertEqualsDouble(c(4), 3.0, eps)
      assertEqualsDouble(c(5), 4.0, eps)
      assertEqualsDouble(c(8), 5.0, eps)
      assertEqualsDouble(c(9), 6.0, eps)
    }
  }
