package palladium.blas

import scala.scalanative.unsafe.*
import scala.scalanative.unsigned.*

/** Minimal CBLAS FFI bindings for the operations needed by BlasExec.
  *
  * We bind only what LowProgram uses:
  *   - Level 1: dscal, dcopy, daxpy, dnrm2, dasum, ddot
  *   - Level 2: dgemv
  *   - Level 3: dgemm
  *
  * Linking against system libcblas (OpenBLAS, ATLAS, or Accelerate).
  */
@link("cblas")
@extern
object Cblas:

  // ── CBLAS enums ──────────────────────────────────────────────────────

  type CBLAS_ORDER     = CInt
  type CBLAS_TRANSPOSE = CInt

  // ── Level 1: Vector operations ───────────────────────────────────────

  /** x := alpha * x */
  def cblas_dscal(N: CInt, alpha: CDouble, X: Ptr[CDouble], incX: CInt): Unit =
    extern

  /** y := x  (copy) */
  def cblas_dcopy(
      N: CInt,
      X: Ptr[CDouble],
      incX: CInt,
      Y: Ptr[CDouble],
      incY: CInt
  ): Unit = extern

  /** y := alpha * x + y */
  def cblas_daxpy(
      N: CInt,
      alpha: CDouble,
      X: Ptr[CDouble],
      incX: CInt,
      Y: Ptr[CDouble],
      incY: CInt
  ): Unit = extern

  /** L2 norm: ||x||_2 */
  def cblas_dnrm2(N: CInt, X: Ptr[CDouble], incX: CInt): CDouble = extern

  /** L1 norm: sum(|x_i|) */
  def cblas_dasum(N: CInt, X: Ptr[CDouble], incX: CInt): CDouble = extern

  /** dot product: x . y */
  def cblas_ddot(
      N: CInt,
      X: Ptr[CDouble],
      incX: CInt,
      Y: Ptr[CDouble],
      incY: CInt
  ): CDouble = extern

  // ── Level 2: Matrix-vector operations ────────────────────────────────

  /** y := alpha * op(A) * x + beta * y
    *
    * op(A) = A if trans == CblasNoTrans, A^T if trans == CblasTrans
    */
  def cblas_dgemv(
      order: CBLAS_ORDER,
      trans: CBLAS_TRANSPOSE,
      M: CInt,
      N: CInt,
      alpha: CDouble,
      A: Ptr[CDouble],
      lda: CInt,
      X: Ptr[CDouble],
      incX: CInt,
      beta: CDouble,
      Y: Ptr[CDouble],
      incY: CInt
  ): Unit = extern

  // ── Level 3: Matrix-matrix operations ────────────────────────────────

  /** C := alpha * op(A) * op(B) + beta * C
    *
    * General matrix multiply. This is the workhorse for our MatMul lowering.
    */
  def cblas_dgemm(
      order: CBLAS_ORDER,
      transA: CBLAS_TRANSPOSE,
      transB: CBLAS_TRANSPOSE,
      M: CInt,
      N: CInt,
      K: CInt,
      alpha: CDouble,
      A: Ptr[CDouble],
      lda: CInt,
      B: Ptr[CDouble],
      ldb: CInt,
      beta: CDouble,
      C: Ptr[CDouble],
      ldc: CInt
  ): Unit = extern

/** CBLAS enum constants. Separated from @extern object because Scala Native
  * requires @extern objects to only contain extern method declarations.
  */
object CblasConstants:
  final val CblasRowMajor: CInt = 101
  final val CblasColMajor: CInt = 102
  final val CblasNoTrans: CInt  = 111
  final val CblasTrans: CInt    = 112
