package palladium.ein

import palladium.NumberLike
import scala.reflect.ClassTag

enum Block[A]:
  case Atom(factory: (Ein[A], String, Option[String]) => Ein[A])
  case Sequence(blocks: Vector[Block[A]])
  case Repeat(block: Block[A], n: Int)
  case Residual(block: Block[A])

  def >>(that: Block[A]): Block[A] = (this, that) match
    case (Sequence(bs1), Sequence(bs2)) => Sequence(bs1 ++ bs2)
    case (Sequence(bs1), _)             => Sequence(bs1 :+ that)
    case (_, Sequence(bs2))             => Sequence(this +: bs2)
    case _                              => Sequence(Vector(this, that))

  def *(n: Int): Block[A] = Repeat(this, n)

  def residual: Block[A] = Residual(this)

  def materialize(input: Ein[A], prefix: String = "block"): Ein[A] =
    val counter = new Block.Counter()
    build(input, prefix, counter, None)

  private[ein] def build(
      input: Ein[A],
      prefix: String,
      counter: Block.Counter,
      targetOutDim: Option[String]
  ): Ein[A] =
    this match
      case Atom(factory) =>
        val scope = s"$prefix${counter.next()}"
        factory(input, scope, targetOutDim)

      case Sequence(blocks) =>
        if blocks.isEmpty then input
        else
          val init = blocks.init.foldLeft(input) { (acc, block) =>
            block.build(acc, prefix, counter, None)
          }
          blocks.last.build(init, prefix, counter, targetOutDim)

      case Repeat(block, n) =>
        if n <= 0 then input
        else
          val init = (0 until n - 1).foldLeft(input) { (acc, _) =>
            block.build(acc, prefix, counter, None)
          }
          block.build(init, prefix, counter, targetOutDim)

      case Residual(block) =>
        val inputDimName = input.outputDims.headOption.map(_.name)
        val output = block.build(input, prefix, counter, inputDimName)
        Ein.ElemAdd(input, output)

object Block:

  case class Network[A](input: Ein[A], block: Block[A]):
    def >>(that: Block[A]): Network[A] = Network(input, block >> that)
    def materialize: Ein[A] = block.materialize(input)

  extension [A](input: Ein[A])
    def >>(block: Block[A]): Network[A] = Network(input, block)

  class Counter:
    private var count = 0
    def next(): Int =
      val n = count
      count += 1
      n

  def dense[A: NumberLike: ClassTag](outSize: Int, activation: Activation): Block[A] =
    Atom { (input, scope, targetDimName) =>
      val inputDims = input.outputDims
      val outDimName = targetDimName.getOrElse(s"${scope}_out")
      val outDim = Dim(outDimName, outSize)
      val wDims = List(outDim) ++ inputDims
      val bDims = List(outDim)
      val w = Ein.Param(s"$scope/W", wDims, TensorData.zeros[A](wDims))
      val b = Ein.Param(s"$scope/b", bDims, TensorData.zeros[A](bDims))
      Ein.Activate(activation, (w * input) + b)
    }

  def dense(outSize: Int, activation: Activation, init: EinDsl.Init)(using rng: java.util.Random = java.util.Random()): Block[Double] =
    Atom { (input, scope, targetDimName) =>
      val inputDims = input.outputDims
      val outDimName = targetDimName.getOrElse(s"${scope}_out")
      val outDim = Dim(outDimName, outSize)
      val wDims = List(outDim) ++ inputDims
      val bDims = List(outDim)
      val fanIn = inputDims.map(_.size).product
      val wData = EinDsl.generate(init, wDims.map(_.size).product, fanIn, outSize, rng)
      val w = Ein.Param(s"$scope/W", wDims, TensorData.fromArray(wDims, wData))
      val b = Ein.Param(s"$scope/b", bDims, TensorData.zeros[Double](bDims))
      Ein.Activate(activation, (w * input) + b)
    }

  def linear[A: NumberLike: ClassTag](outSize: Int): Block[A] =
    Atom { (input, scope, targetDimName) =>
      val inputDims = input.outputDims
      val outDimName = targetDimName.getOrElse(s"${scope}_out")
      val outDim = Dim(outDimName, outSize)
      val wDims = List(outDim) ++ inputDims
      val bDims = List(outDim)
      val w = Ein.Param(s"$scope/W", wDims, TensorData.zeros[A](wDims))
      val b = Ein.Param(s"$scope/b", bDims, TensorData.zeros[A](bDims))
      (w * input) + b
    }

  def linear(outSize: Int, init: EinDsl.Init)(using rng: java.util.Random = java.util.Random()): Block[Double] =
    Atom { (input, scope, targetDimName) =>
      val inputDims = input.outputDims
      val outDimName = targetDimName.getOrElse(s"${scope}_out")
      val outDim = Dim(outDimName, outSize)
      val wDims = List(outDim) ++ inputDims
      val bDims = List(outDim)
      val fanIn = inputDims.map(_.size).product
      val wData = EinDsl.generate(init, wDims.map(_.size).product, fanIn, outSize, rng)
      val w = Ein.Param(s"$scope/W", wDims, TensorData.fromArray(wDims, wData))
      val b = Ein.Param(s"$scope/b", bDims, TensorData.zeros[Double](bDims))
      (w * input) + b
    }

  def activate[A](f: Activation): Block[A] =
    Atom { (input, _, _) => Ein.Activate(f, input) }

  def fn[A](f: Ein[A] => Ein[A]): Block[A] =
    Atom { (input, _, _) => f(input) }

  def gelu[A]: Block[A] = activate(Activation.GELU)

  def swish[A]: Block[A] = activate(Activation.Swish)

  def softmax[A](overDim: String): Block[A] =
    Atom { (input, _, _) => Ein.Softmax(input, overDim) }

  def logSoftmax[A](overDim: String): Block[A] =
    Atom { (input, _, _) => Ein.LogSoftmax(input, overDim) }

  def layerNorm[A: NumberLike: ClassTag](overDims: List[String]): Block[A] =
    Atom { (input, scope, _) =>
      val normDims = overDims.flatMap(name => input.outputDims.find(_.name == name))
      val scale = Ein.Param(s"$scope/scale", normDims, TensorData.fill[A](normDims, summon[NumberLike[A]].fromInt(1)))
      val bias = Ein.Param(s"$scope/bias", normDims, TensorData.zeros[A](normDims))
      Ein.LayerNorm(input, scale, bias, overDims, 1e-5)
    }

  def embedding[A: NumberLike: ClassTag](
      vocabSize: Int,
      dim: Int,
      indices: TensorData[Int],
      vocabDimName: String = "vocab",
      embedDimName: String = "embed"
  ): Block[A] =
    Atom { (_, scope, _) =>
      val vocabDim = Dim(vocabDimName, vocabSize)
      val embedDim = Dim(embedDimName, dim)
      val tableDims = List(vocabDim, embedDim)
      val table = Ein.Param(s"$scope/embedding", tableDims, TensorData.zeros[A](tableDims))
      Ein.Gather(table, indices, vocabDimName)
    }

  // ── Transformer building blocks ─────────────────────────────────────

  /** Scaled dot-product self-attention (single head).
    *
    * Input: [seq, model]
    * Output: [seq, model] (via output projection)
    *
    * Internally:
    *   Q = W_q × input           → [qd, seq]      (contracts over model)
    *   K = W_k × input_kv        → [qd, kSeq]     (contracts over model)
    *   V = W_v × input_kv        → [vd, kSeq]     (contracts over model)
    *   scores = Q × K            → [seq, kSeq]     (contracts over qd)
    *   attn = softmax(scores + mask, kSeq) × V → [seq, vd]
    *   output = W_o × attn       → [model, seq]
    *
    * Key/value input has seq renamed to "kSeq" via Reshape so that
    * Q × K contracts over the attention dim, not the sequence dim.
    */
  def attention[A: NumberLike: ClassTag](
      headDim: Int,
      mask: Option[TensorData[A]] = None,
      seqDimName: String = "seq",
      modelDimName: String = "model"
  ): Block[A] =
    Atom { (input, scope, _) =>
      val num = summon[NumberLike[A]]
      val inputDims = input.outputDims
      val seqDim = inputDims.find(_.name == seqDimName).get
      val modelDim = inputDims.find(_.name == modelDimName).get
      val seqLen = seqDim.size

      val qdDim = Dim(s"${scope}_qd", headDim)
      val vdDim = Dim(s"${scope}_vd", headDim)
      val kSeqDim = Dim("kSeq", seqLen)

      // Rename input's seq → kSeq for key/value
      val kvDims = inputDims.map(d => if d.name == seqDimName then kSeqDim else d)
      val inputKV = Ein.Reshape(input, kvDims)

      // Projections
      val wQ = Ein.Param(s"$scope/W_q", List(qdDim, modelDim), TensorData.zeros[A](List(qdDim, modelDim)))
      val wK = Ein.Param(s"$scope/W_k", List(qdDim, modelDim), TensorData.zeros[A](List(qdDim, modelDim)))
      val wV = Ein.Param(s"$scope/W_v", List(vdDim, modelDim), TensorData.zeros[A](List(vdDim, modelDim)))
      val wO = Ein.Param(s"$scope/W_o", List(modelDim, vdDim), TensorData.zeros[A](List(modelDim, vdDim)))

      val q = wQ * input      // [qdDim, seq]
      val k = wK * inputKV    // [qdDim, kSeq]
      val v = wV * inputKV    // [vdDim, kSeq]

      // Scaled attention scores: Q × K / sqrt(headDim)
      val scores = q * k       // [seq, kSeq] — contracts over qdDim
      val scale = num.div(num.fromInt(1), num.sqrt(num.fromInt(headDim)))
      val scaleTensor = Ein.Fill(scale, scores.outputDims)
      val scaledScores = Ein.ElemMul(scores, scaleTensor)

      // Apply causal mask (if provided)
      val maskedScores = mask match
        case Some(maskData) =>
          val maskParam = Ein.Param(s"$scope/mask", maskData.dims, maskData)
          Ein.ElemAdd(scaledScores, maskParam)
        case None => scaledScores

      // Softmax over key positions
      val attnWeights = Ein.Softmax(maskedScores, "kSeq")

      // Weighted sum of values
      val attnOutput = attnWeights * v  // [seq, vdDim] — contracts over kSeq

      // Output projection
      wO * attnOutput  // [model, seq] — contracts over vdDim
    }

  /** Multi-head self-attention.
    *
    * Each head computes independent attention with its own Q/K/V/O
    * projections of size headDim, then all heads' outputs (projected
    * back to model dim) are summed.
    *
    * This is equivalent to standard multi-head attention with a
    * block-diagonal output projection matrix.
    */
  def multiHeadAttention[A: NumberLike: ClassTag](
      nHeads: Int,
      headDim: Int,
      mask: Option[TensorData[A]] = None,
      seqDimName: String = "seq",
      modelDimName: String = "model"
  ): Block[A] =
    Atom { (input, scope, _) =>
      val num = summon[NumberLike[A]]
      val inputDims = input.outputDims
      val seqDim = inputDims.find(_.name == seqDimName).get
      val modelDim = inputDims.find(_.name == modelDimName).get
      val seqLen = seqDim.size

      val kSeqDim = Dim("kSeq", seqLen)
      val kvDims = inputDims.map(d => if d.name == seqDimName then kSeqDim else d)
      val inputKV = Ein.Reshape(input, kvDims)

      // Each head computes attention independently and projects back to model dim
      val headOutputs = (0 until nHeads).map { h =>
        val qdDim = Dim(s"${scope}_h${h}_qd", headDim)
        val vdDim = Dim(s"${scope}_h${h}_vd", headDim)

        val wQ = Ein.Param(s"$scope/h$h/W_q", List(qdDim, modelDim),
          TensorData.zeros[A](List(qdDim, modelDim)))
        val wK = Ein.Param(s"$scope/h$h/W_k", List(qdDim, modelDim),
          TensorData.zeros[A](List(qdDim, modelDim)))
        val wV = Ein.Param(s"$scope/h$h/W_v", List(vdDim, modelDim),
          TensorData.zeros[A](List(vdDim, modelDim)))
        val wO = Ein.Param(s"$scope/h$h/W_o", List(modelDim, vdDim),
          TensorData.zeros[A](List(modelDim, vdDim)))

        val q = wQ * input
        val k = wK * inputKV
        val v = wV * inputKV

        val scores = q * k
        val scale = num.div(num.fromInt(1), num.sqrt(num.fromInt(headDim)))
        val scaleTensor = Ein.Fill(scale, scores.outputDims)
        val scaledScores = Ein.ElemMul(scores, scaleTensor)

        val maskedScores = mask match
          case Some(maskData) =>
            val maskParam = Ein.Param(s"$scope/h$h/mask", maskData.dims, maskData)
            Ein.ElemAdd(scaledScores, maskParam)
          case None => scaledScores

        val attnWeights = Ein.Softmax(maskedScores, "kSeq")
        val attnOutput = attnWeights * v
        wO * attnOutput
      }

      // Sum all head outputs
      headOutputs.reduce[Ein[A]](Ein.ElemAdd(_, _))
    }

  /** Position-wise feed-forward network (two linear layers with GELU).
    *
    * Input: [seq, model]
    * Output: [seq, model]
    *
    * FFN(x) = W2 × GELU(W1 × x + b1) + b2
    */
  def feedForward[A: NumberLike: ClassTag](
      ffDim: Int,
      activation: Activation = Activation.GELU,
      seqDimName: String = "seq",
      modelDimName: String = "model"
  ): Block[A] =
    Atom { (input, scope, _) =>
      val inputDims = input.outputDims
      val modelDim = inputDims.find(_.name == modelDimName).get

      val hidDim = Dim(s"${scope}_ff", ffDim)

      val w1 = Ein.Param(s"$scope/ff/W1", List(hidDim, modelDim),
        TensorData.zeros[A](List(hidDim, modelDim)))
      val b1 = Ein.Param(s"$scope/ff/b1", List(hidDim),
        TensorData.zeros[A](List(hidDim)))
      val w2 = Ein.Param(s"$scope/ff/W2", List(modelDim, hidDim),
        TensorData.zeros[A](List(modelDim, hidDim)))
      val b2 = Ein.Param(s"$scope/ff/b2", List(modelDim),
        TensorData.zeros[A](List(modelDim)))

      val hidden = Ein.Activate(activation, (w1 * input) + b1)
      (w2 * hidden) + b2
    }

  /** Transformer block (pre-norm architecture).
    *
    * x → LayerNorm → MultiHeadAttention → + (residual) →
    *   → LayerNorm → FeedForward → + (residual)
    */
  def transformerBlock[A: NumberLike: ClassTag](
      nHeads: Int,
      headDim: Int,
      ffDim: Int,
      seqLen: Int,
      activation: Activation = Activation.GELU,
      seqDimName: String = "seq",
      modelDimName: String = "model"
  ): Block[A] =
    val mask = Attention.causalMask[A](seqLen, seqDimName, "kSeq")
    Atom { (input, scope, _) =>
      val inputDims = input.outputDims
      val modelDim = inputDims.find(_.name == modelDimName).get

      // Sub-block 1: LayerNorm → MHA → Residual
      val ln1Dims = List(modelDimName)
      val ln1NormDims = ln1Dims.flatMap(name => inputDims.find(_.name == name))
      val ln1Scale = Ein.Param(s"$scope/ln1/scale", ln1NormDims,
        TensorData.fill[A](ln1NormDims, summon[NumberLike[A]].fromInt(1)))
      val ln1Bias = Ein.Param(s"$scope/ln1/bias", ln1NormDims,
        TensorData.zeros[A](ln1NormDims))
      val normed1 = Ein.LayerNorm(input, ln1Scale, ln1Bias, ln1Dims, 1e-5)

      // Build attention inline (need unique scope per head)
      val attnBlock = multiHeadAttention[A](nHeads, headDim, Some(mask), seqDimName, modelDimName)
      val attnCounter = new Counter()
      val attnOut = attnBlock.build(normed1, s"${scope}/attn", attnCounter, None)
      val afterAttn = Ein.ElemAdd(input, attnOut)

      // Sub-block 2: LayerNorm → FFN → Residual
      val ln2NormDims = ln1Dims.flatMap(name => afterAttn.outputDims.find(_.name == name))
      val ln2Scale = Ein.Param(s"$scope/ln2/scale", ln2NormDims,
        TensorData.fill[A](ln2NormDims, summon[NumberLike[A]].fromInt(1)))
      val ln2Bias = Ein.Param(s"$scope/ln2/bias", ln2NormDims,
        TensorData.zeros[A](ln2NormDims))
      val normed2 = Ein.LayerNorm(afterAttn, ln2Scale, ln2Bias, ln1Dims, 1e-5)

      val ffBlock = feedForward[A](ffDim, activation, seqDimName, modelDimName)
      val ffCounter = new Counter()
      val ffOut = ffBlock.build(normed2, s"${scope}/ff", ffCounter, None)
      Ein.ElemAdd(afterAttn, ffOut)
    }

  /** Full decoder-only transformer stack.
    *
    * N layers of: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual
    */
  def transformer[A: NumberLike: ClassTag](
      nLayers: Int,
      nHeads: Int,
      headDim: Int,
      ffDim: Int,
      seqLen: Int,
      activation: Activation = Activation.GELU,
      seqDimName: String = "seq",
      modelDimName: String = "model"
  ): Block[A] =
    val blocks = (0 until nLayers).map { i =>
      transformerBlock[A](nHeads, headDim, ffDim, seqLen, activation, seqDimName, modelDimName)
    }.toVector
    Sequence(blocks)
