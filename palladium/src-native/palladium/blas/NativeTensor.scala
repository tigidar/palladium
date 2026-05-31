package palladium.blas

import scala.scalanative.unsafe.*

/** A tensor backed by a raw Ptr[Double] with shape metadata.
  *
  * Allocated from a Zone (arena). No GC pressure, no boxing.
  * The pointer is valid for the lifetime of the enclosing Zone.
  *
  * All operations are in-place mutations on the underlying buffer.
  */
final class NativeTensor private (
    val data: Ptr[Double],
    val shape: List[Int],
    val totalSize: Int
):

  /** Read element at flat index. */
  inline def apply(i: Int): Double = data(i)

  /** Write element at flat index. */
  inline def update(i: Int, v: Double): Unit = data(i) = v

  /** Zero all elements. */
  def zero(): Unit =
    var i = 0
    while i < totalSize do
      data(i) = 0.0
      i += 1

  /** Fill all elements with a constant. */
  def fill(v: Double): Unit =
    var i = 0
    while i < totalSize do
      data(i) = v
      i += 1

  /** Copy contents from another tensor (sizes must match). */
  def copyFrom(other: NativeTensor): Unit =
    val n = totalSize.min(other.totalSize)
    Cblas.cblas_dcopy(n, other.data, 1, data, 1)

  /** Extract data to a Scala Array[Double] (for interop / testing). */
  def toArray: Array[Double] =
    val arr = new Array[Double](totalSize)
    var i = 0
    while i < totalSize do
      arr(i) = data(i)
      i += 1
    arr

object NativeTensor:

  /** Allocate a zero-initialized tensor from the given Zone (arena). */
  def zeros(shape: List[Int])(using zone: Zone): NativeTensor =
    val size = if shape.isEmpty then 1 else shape.product
    val ptr = alloc[Double](size)
    val t = new NativeTensor(ptr, shape, size)
    t.zero()
    t

  /** Allocate a tensor filled with ones. */
  def ones(shape: List[Int])(using Zone): NativeTensor =
    val t = zeros(shape)
    t.fill(1.0)
    t

  /** Allocate a tensor from an existing Array[Double]. */
  def fromArray(shape: List[Int], arr: Array[Double])(using Zone): NativeTensor =
    val size = if shape.isEmpty then 1 else shape.product
    val ptr = alloc[Double](size)
    val t = new NativeTensor(ptr, shape, size)
    var i = 0
    val n = size.min(arr.length)
    while i < n do
      ptr(i) = arr(i)
      i += 1
    // zero remaining if arr is shorter
    while i < size do
      ptr(i) = 0.0
      i += 1
    t

  /** Allocate and fill with a constant value. */
  def filled(shape: List[Int], value: Double)(using Zone): NativeTensor =
    val t = zeros(shape)
    t.fill(value)
    t
