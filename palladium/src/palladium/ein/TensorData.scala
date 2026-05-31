package palladium.ein

import palladium.NumberLike
import scala.reflect.ClassTag

case class TensorData[A](data: Array[A], dims: List[Dim], strides: List[Int]):

  def size: Int = data.length

  def offset(indices: Array[Int]): Int =
    var off = 0
    var i = 0
    while i < indices.length do
      off += indices(i) * strides(i)
      i += 1
    off

  def apply(indices: Array[Int]): A = data(offset(indices))

  def update(indices: Array[Int], value: A): Unit =
    data(offset(indices)) = value

  def toRows: IndexedSeq[IndexedSeq[A]] =
    require(dims.size == 2, s"toRows requires a 2D tensor, got ${dims.size}D")
    val cols = dims(1).size
    data.grouped(cols).map(_.toIndexedSeq).toIndexedSeq

  override def toString: String =
    val dimStr = dims.map(d => s"${d.name}=${d.size}").mkString("[", ", ", "]")
    s"TensorData$dimStr(${data.take(8).mkString(", ")}${if data.length > 8 then ", ..." else ""})"

  override def equals(that: Any): Boolean = that match
    case t: TensorData[?] =>
      dims == t.dims && data.length == t.data.length && {
        var i = 0
        var eq = true
        while i < data.length && eq do
          eq = data(i) == t.data(i)
          i += 1
        eq
      }
    case _ => false

  override def hashCode: Int =
    var h = dims.hashCode * 31
    var i = 0
    while i < data.length do
      h = h * 31 + data(i).hashCode
      i += 1
    h

object TensorData:

  def computeStrides(dims: List[Dim]): List[Int] =
    if dims.isEmpty then Nil
    else
      val sizes = dims.map(_.size)
      val strides = new Array[Int](sizes.length)
      strides(sizes.length - 1) = 1
      var i = sizes.length - 2
      while i >= 0 do
        strides(i) = strides(i + 1) * sizes(i + 1)
        i -= 1
      strides.toList

  def zeros[A: NumberLike: ClassTag](dims: List[Dim]): TensorData[A] =
    val num = summon[NumberLike[A]]
    val totalSize = if dims.isEmpty then 1 else dims.map(_.size).product
    val data = Array.fill(totalSize)(num.fromInt(0))
    TensorData(data, dims, computeStrides(dims))

  def fill[A: ClassTag](dims: List[Dim], value: A): TensorData[A] =
    val totalSize = if dims.isEmpty then 1 else dims.map(_.size).product
    val data = Array.fill(totalSize)(value)
    TensorData(data, dims, computeStrides(dims))

  def fromArray[A: ClassTag](dims: List[Dim], data: Array[A]): TensorData[A] =
    TensorData(data.clone(), dims, computeStrides(dims))

  def scalar[A: ClassTag](value: A): TensorData[A] =
    TensorData(Array(value), Nil, Nil)
