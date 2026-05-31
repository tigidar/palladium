package palladium.ein

import palladium.NumberLike
import sourcecode.Name
import scala.reflect.ClassTag

object EinDsl:

  // --- Initializers ---

  enum Init:
    case Uniform(low: Double, high: Double)
    case Xavier
    case Zeros

  def uniform(low: Double, high: Double): Init = Init.Uniform(low, high)
  val xavier: Init = Init.Xavier
  val zeros: Init = Init.Zeros

  private[ein] def generate(init: Init, size: Int, fanIn: Int, fanOut: Int, rng: java.util.Random): Array[Double] =
    init match
      case Init.Uniform(low, high) =>
        Array.fill(size)(low + (high - low) * rng.nextDouble())
      case Init.Xavier =>
        val limit = Math.sqrt(6.0 / (fanIn + fanOut))
        Array.fill(size)(-limit + 2 * limit * rng.nextDouble())
      case Init.Zeros =>
        new Array[Double](size)

  // --- Dimension factory ---

  extension (size: Int)
    def dim(using name: Name): Dim = Dim(name.value, size)

  // --- Tensor factories ---

  def input[A](dims: Dim*)(using name: Name): Ein[A] =
    Ein.Input(name.value, dims.toList)

  def weight(mapping: (Dim, Dim), values: Double*)(using name: Name): Ein[Double] =
    val (from, to) = mapping
    val dims = List(to, from)
    Ein.Param(name.value, dims, TensorData.fromArray(dims, values.toArray))

  def weight(mapping: (Dim, Dim), init: Init)(using name: Name, rng: java.util.Random = java.util.Random()): Ein[Double] =
    val (from, to) = mapping
    val dims = List(to, from)
    val size = to.size * from.size
    Ein.Param(name.value, dims, TensorData.fromArray(dims, generate(init, size, from.size, to.size, rng)))

  def bias(dim: Dim, values: Double*)(using name: Name): Ein[Double] =
    Ein.Param(name.value, List(dim), TensorData.fromArray(List(dim), values.toArray))

  def bias(dim: Dim, init: Init)(using name: Name, rng: java.util.Random = java.util.Random()): Ein[Double] =
    val dims = List(dim)
    Ein.Param(name.value, dims, TensorData.fromArray(dims, generate(init, dim.size, dim.size, dim.size, rng)))

  extension (input: Ein[Double])
    def data(values: Double*): Map[String, TensorData[Double]] = input match
      case Ein.Input(id, dims) =>
        Map(id -> TensorData.fromArray(dims, values.toArray))
      case other =>
        throw IllegalArgumentException(s"data(...) can only be called on Ein.Input, got: $other")

  def relu[A](expr: Ein[A]): Ein[A] = Ein.Activate(Activation.ReLU, expr)
  def sigmoid[A](expr: Ein[A]): Ein[A] = Ein.Activate(Activation.Sigmoid, expr)
  def tanh[A](expr: Ein[A]): Ein[A] = Ein.Activate(Activation.Tanh, expr)
  def gelu[A](expr: Ein[A]): Ein[A] = Ein.Activate(Activation.GELU, expr)
  def swish[A](expr: Ein[A]): Ein[A] = Ein.Activate(Activation.Swish, expr)
  def softmax[A](expr: Ein[A], overDim: String): Ein[A] = Ein.Softmax(expr, overDim)
  def logSoftmax[A](expr: Ein[A], overDim: String): Ein[A] = Ein.LogSoftmax(expr, overDim)
  def layerNorm[A](expr: Ein[A], scale: Ein[A], bias: Ein[A], overDims: List[String], eps: Double = 1e-5): Ein[A] =
    Ein.LayerNorm(expr, scale, bias, overDims, eps)

  extension [A](expr: Ein[A])
    def sumOver(dims: String*): Ein[A] = Ein.ReduceSum(expr, dims.toList)
    def sum: Ein[A] = Ein.ReduceSum(expr, expr.outputDims.map(_.name))

    def parameters: Map[String, TensorData[A]] =
      def collect(e: Ein[A]): Map[String, TensorData[A]] = e match
        case Ein.Param(id, _, data)           => Map(id -> data)
        case Ein.Input(_, _)                  => Map.empty
        case Ein.Fill(_, _)                   => Map.empty
        case Ein.Ones(_)                      => Map.empty
        case Ein.Zeros(_)                     => Map.empty
        case Ein.Contract(l, r)               => collect(l) ++ collect(r)
        case Ein.ElemAdd(l, r)                => collect(l) ++ collect(r)
        case Ein.ElemSub(l, r)                => collect(l) ++ collect(r)
        case Ein.ElemMul(l, r)                => collect(l) ++ collect(r)
        case Ein.Activate(_, arg)             => collect(arg)
        case Ein.ActivateDeriv(_, arg)        => collect(arg)
        case Ein.ReduceSum(arg, _)            => collect(arg)
        case Ein.Broadcast(arg, _)            => collect(arg)
        case Ein.Transpose(arg, _)            => collect(arg)
        case Ein.Softmax(arg, _)              => collect(arg)
        case Ein.LogSoftmax(arg, _)           => collect(arg)
        case Ein.LayerNorm(arg, s, b, _, _)   => collect(arg) ++ collect(s) ++ collect(b)
        case Ein.Reshape(arg, _)              => collect(arg)
        case Ein.Slice(arg, _, _, _)          => collect(arg)
        case Ein.Gather(table, _, _)          => collect(table)
        case Ein.Scatter(src, _, _, _)        => collect(src)
      collect(expr)
    def square: Ein[A] = Ein.ElemMul(expr, expr)
    def **(n: Int): Ein[A] =
      require(n >= 1, s"** requires n >= 1, got $n")
      (1 until n).foldLeft(expr)((acc, _) => Ein.ElemMul(acc, expr))

    def broadcast(targetDims: Dim*): Ein[A] = Ein.Broadcast(expr, targetDims.toList)
    def t(perm: String*): Ein[A] = Ein.Transpose(expr, perm.toList)

    @annotation.targetName("softmaxExt")
    def softmax(overDim: String): Ein[A] = Ein.Softmax(expr, overDim)

    @annotation.targetName("logSoftmaxExt")
    def logSoftmax(overDim: String): Ein[A] = Ein.LogSoftmax(expr, overDim)

  extension [A: NumberLike: ClassTag](logits: Ein[A])
    def crossEntropyLoss(targets: Ein[A], classDim: String): Ein[A] =
      EinLoss.crossEntropy(logits, targets, classDim)

  object syntax:
    private type Feed[A] = Map[String, TensorData[A]]

    extension [A: NumberLike: ClassTag](expr: Ein[A])
      def eval(): TensorData[A] =
        EinEval.eval(expr, Map.empty)
      def eval(input: Feed[A]): TensorData[A] =
        EinEval.eval(expr, input)
      def eval(input: Feed[A], target: Feed[A]): TensorData[A] =
        EinEval.eval(expr, input ++ target)

      def backward(): Feed[A] =
        EinGrad.backward(expr, Map.empty)
      def backward(input: Feed[A]): Feed[A] =
        EinGrad.backward(expr, input)
      def backward(input: Feed[A], target: Feed[A]): Feed[A] =
        EinGrad.backward(expr, input ++ target)
