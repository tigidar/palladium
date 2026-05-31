package palladium.ein

import palladium.NumberLike
import scala.reflect.ClassTag

object PositionalEncoding:

  /** Sinusoidal positional encoding (Vaswani et al., "Attention Is All You Need").
    *
    * PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
    * PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    *
    * Returns a TensorData of shape [Dim(posDimName, maxLen), Dim(embedDimName, dim)].
    */
  def sinusoidal[A: NumberLike: ClassTag](
      maxLen: Int,
      dim: Int,
      posDimName: String = "seq",
      embedDimName: String = "embed"
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    val posDim = Dim(posDimName, maxLen)
    val embedDim = Dim(embedDimName, dim)
    val dims = List(posDim, embedDim)
    val data = new Array[A](maxLen * dim)

    var pos = 0
    while pos < maxLen do
      var i = 0
      while i < dim do
        val divTerm = math.pow(10000.0, (i / 2) * 2.0 / dim)
        val angle = approxFromDouble[A](pos.toDouble / divTerm)
        data(pos * dim + i) =
          if i % 2 == 0 then num.sin(angle)
          else num.cos(angle)
        i += 1
      pos += 1

    TensorData.fromArray(dims, data)

  private def approxFromDouble[A: NumberLike](d: Double): A =
    val num = summon[NumberLike[A]]
    if d == 0.0 then num.fromInt(0)
    else
      val scale = 1000000
      val scaled = math.round(d * scale).toInt
      if scaled == 0 then num.fromInt(0)
      else num.div(num.fromInt(scaled), num.fromInt(scale))
