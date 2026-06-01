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
    case Ones

  def uniform(low: Double, high: Double): Init = Init.Uniform(low, high)
  val xavier: Init = Init.Xavier
  val zeros: Init = Init.Zeros
  val ones: Init = Init.Ones

  private[ein] def generate(init: Init, size: Int, fanIn: Int, fanOut: Int, rng: java.util.Random): Array[Double] =
    init match
      case Init.Uniform(low, high) =>
        Array.fill(size)(low + (high - low) * rng.nextDouble())
      case Init.Xavier =>
        val limit = Math.sqrt(6.0 / (fanIn + fanOut))
        Array.fill(size)(-limit + 2 * limit * rng.nextDouble())
      case Init.Zeros =>
        new Array[Double](size)
      case Init.Ones =>
        Array.fill(size)(1.0)

  // --- Scope: an ambient parameter namespace for reusable layers ---
  //
  // `sourcecode.Name` gives a leaf factory its *binding* name ("w", "b", "q").
  // That is all you need in straight-line code, but a reusable layer is stamped
  // out many times and each instance needs a *distinct* parameter id. An ambient
  // `Scope` supplies the prefix: the leaf still comes from the binding, the
  // prefix comes from the scope, and the two combine into "layer0.attn.h0.q.w".
  //
  // Straight-line `val`s get their discriminator for free (each `val` has its
  // own name); loops have no per-iteration binding, so they push an explicit
  // index with `scoped(s"h$h") { ... }`.
  opaque type Scope = String

  object Scope:
    /** The empty (top-level) scope — leaf ids are then just the binding name. */
    val root: Scope = ""
    def apply(path: String): Scope = path

  extension (scope: Scope)
    /** Push a path segment, dot-separated. */
    def /(child: String): Scope = if scope.isEmpty then child else s"$scope.$child"
    def path: String = scope

  given Scope = Scope.root

  /** Run `body` with `name` pushed onto the ambient scope. */
  def scoped[A](name: String)(body: Scope ?=> A)(using parent: Scope): A =
    body(using parent / name)

  private def scopedId(leaf: String)(using scope: Scope): String = (scope / leaf).path

  // --- Dimension factory ---

  extension (size: Int)
    def dim(using name: Name): Dim = Dim(name.value, size)

  // --- Tensor factories ---

  def input[A](dims: Dim*)(using name: Name, scope: Scope): Ein[A] =
    Ein.Input(scopedId(name.value), dims.toList)

  def weight(mapping: (Dim, Dim), values: Double*)(using name: Name, scope: Scope): Ein[Double] =
    val (from, to) = mapping
    val dims = List(to, from)
    Ein.Param(scopedId(name.value), dims, TensorData.fromArray(dims, values.toArray))

  def weight(mapping: (Dim, Dim), init: Init)(using name: Name, scope: Scope, rng: java.util.Random = java.util.Random()): Ein[Double] =
    val (from, to) = mapping
    val dims = List(to, from)
    val size = to.size * from.size
    Ein.Param(scopedId(name.value), dims, TensorData.fromArray(dims, generate(init, size, from.size, to.size, rng)))

  def bias(dim: Dim, values: Double*)(using name: Name, scope: Scope): Ein[Double] =
    Ein.Param(scopedId(name.value), List(dim), TensorData.fromArray(List(dim), values.toArray))

  def bias(dim: Dim, init: Init)(using name: Name, scope: Scope, rng: java.util.Random = java.util.Random()): Ein[Double] =
    val dims = List(dim)
    Ein.Param(scopedId(name.value), dims, TensorData.fromArray(dims, generate(init, dim.size, dim.size, dim.size, rng)))

  /** A general parameter of any rank, named from its binding and scope. */
  def param(dims: Dim*)(init: Init)(using name: Name, scope: Scope, rng: java.util.Random = java.util.Random()): Ein[Double] =
    val ds = dims.toList
    val size = if ds.isEmpty then 1 else ds.map(_.size).product
    val fanOut = ds.headOption.map(_.size).getOrElse(1)
    val fanIn = ds.lastOption.map(_.size).getOrElse(1)
    Ein.Param(scopedId(name.value), ds, TensorData.fromArray(ds, generate(init, size, fanIn, fanOut, rng)))

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

    // --- Dim-typed overloads: reference an axis by its value, not a string ---

    @annotation.targetName("sumOverDims")
    def sumOver(dims: Dim*): Ein[A] = Ein.ReduceSum(expr, dims.toList.map(_.name))

    @annotation.targetName("transposeDims")
    def t(perm: Dim*): Ein[A] = Ein.Transpose(expr, perm.toList.map(_.name))

    @annotation.targetName("softmaxDim")
    def softmax(over: Dim): Ein[A] = Ein.Softmax(expr, over.name)

    @annotation.targetName("logSoftmaxDim")
    def logSoftmax(over: Dim): Ein[A] = Ein.LogSoftmax(expr, over.name)

    /** Rename axes by value — a zero-cost `Reshape` to identical sizes under new names. */
    def renameDim(mapping: (Dim, Dim)*): Ein[A] =
      mapping.foreach: (from, to) =>
        require(from.size == to.size, s"renameDim: ${from.name}(${from.size}) -> ${to.name}(${to.size}) must preserve size")
      val byName = mapping.map((from, to) => from.name -> to).toMap
      Ein.Reshape(expr, expr.outputDims.map(d => byName.getOrElse(d.name, d)))

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
