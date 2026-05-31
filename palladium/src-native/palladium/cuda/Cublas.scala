package palladium.cuda

import scala.scalanative.unsafe.*

/** Minimal cuBLAS FFI bindings.
  *
  * cuBLAS is column-major by default (unlike CBLAS which supports row-major).
  * We handle the row→column translation in CudaCompile by transposing the operation:
  *   row-major: y = A * x  →  column-major: y = A^T * x (with swapped M/N)
  */
@link("cublas")
@extern
object Cublas:

  /** Opaque cuBLAS handle. */
  type CublasHandle = Ptr[Byte]
  type CublasStatus = CInt
  type CublasOperation = CInt

  /** Create a cuBLAS context. */
  def cublasCreate_v2(handle: Ptr[CublasHandle]): CublasStatus = extern

  /** Destroy a cuBLAS context. */
  def cublasDestroy_v2(handle: CublasHandle): CublasStatus = extern

  /** y := alpha * op(A) * x + beta * y
    *
    * Column-major matrix-vector multiply on device pointers.
    * A is m×n in column-major storage.
    */
  def cublasDgemv_v2(
      handle: CublasHandle,
      trans: CublasOperation,
      m: CInt,
      n: CInt,
      alpha: Ptr[CDouble],
      A: Ptr[CDouble],
      lda: CInt,
      x: Ptr[CDouble],
      incx: CInt,
      beta: Ptr[CDouble],
      y: Ptr[CDouble],
      incy: CInt
  ): CublasStatus = extern

  /** y := alpha * x + y  (on device) */
  def cublasDaxpy_v2(
      handle: CublasHandle,
      n: CInt,
      alpha: Ptr[CDouble],
      x: Ptr[CDouble],
      incx: CInt,
      y: Ptr[CDouble],
      incy: CInt
  ): CublasStatus = extern

  /** Copy x to y on device. */
  def cublasDcopy_v2(
      handle: CublasHandle,
      n: CInt,
      x: Ptr[CDouble],
      incx: CInt,
      y: Ptr[CDouble],
      incy: CInt
  ): CublasStatus = extern

  /** Scale: x := alpha * x */
  def cublasDscal_v2(
      handle: CublasHandle,
      n: CInt,
      alpha: Ptr[CDouble],
      x: Ptr[CDouble],
      incx: CInt
  ): CublasStatus = extern

object CublasConstants:
  final val CUBLAS_STATUS_SUCCESS: CInt   = 0
  final val CUBLAS_OP_N: CInt             = 0   // No transpose
  final val CUBLAS_OP_T: CInt             = 1   // Transpose
