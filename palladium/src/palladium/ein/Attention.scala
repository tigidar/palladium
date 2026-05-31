package palladium.ein

import palladium.NumberLike
import scala.reflect.ClassTag

object Attention:

  /** Generate a causal (autoregressive) attention mask.
    *
    * Returns a TensorData of shape [Dim(seqDimName, seqLen), Dim(kSeqDimName, seqLen)]
    * where:
    *   - mask(i, j) = 0.0       if i >= j  (token i can attend to token j)
    *   - mask(i, j) = -infinity  if i < j   (token i cannot attend to future token j)
    *
    * This is added to attention scores before softmax, effectively zeroing
    * out attention to future positions.
    */
  def causalMask[A: NumberLike: ClassTag](
      seqLen: Int,
      seqDimName: String = "seq",
      kSeqDimName: String = "kSeq"
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    val seqDim = Dim(seqDimName, seqLen)
    val kSeqDim = Dim(kSeqDimName, seqLen)
    val dims = List(seqDim, kSeqDim)
    val data = new Array[A](seqLen * seqLen)

    var i = 0
    while i < seqLen do
      var j = 0
      while j < seqLen do
        data(i * seqLen + j) =
          if i >= j then num.fromInt(0)
          else num.negativeInfinity
        j += 1
      i += 1

    TensorData.fromArray(dims, data)
