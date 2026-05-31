package palladium.ein

enum Ein[A]:
  // Leaves
  case Param(id: String, dims: List[Dim], data: TensorData[A])
  case Input(id: String, dims: List[Dim])
  case Fill(value: A, dims: List[Dim])
  case Ones(dims: List[Dim])
  case Zeros(dims: List[Dim])

  // Einstein contraction — sum over shared named indices
  case Contract(left: Ein[A], right: Ein[A])

  // Element-wise binary
  case ElemAdd(left: Ein[A], right: Ein[A])
  case ElemSub(left: Ein[A], right: Ein[A])
  case ElemMul(left: Ein[A], right: Ein[A])

  // Activations
  case Activate(f: Activation, arg: Ein[A])
  case ActivateDeriv(f: Activation, arg: Ein[A])

  // Reductions & shape
  case ReduceSum(arg: Ein[A], over: List[String])
  case Broadcast(arg: Ein[A], targetDims: List[Dim])
  case Transpose(arg: Ein[A], perm: List[String])

  // Cross-element normalization (NOT element-wise)
  case Softmax(arg: Ein[A], overDim: String)
  case LogSoftmax(arg: Ein[A], overDim: String)
  case LayerNorm(arg: Ein[A], scale: Ein[A], bias: Ein[A], overDims: List[String], eps: Double = 1e-5)

  // Shape manipulation
  case Reshape(arg: Ein[A], targetDims: List[Dim])
  case Slice(arg: Ein[A], dim: String, from: Int, to: Int)

  // Indexing
  case Gather(table: Ein[A], indices: TensorData[Int], lookupDim: String)
  case Scatter(src: Ein[A], indices: TensorData[Int], lookupDim: String, tableDims: List[Dim])

  // Operators
  def ×(right: Ein[A]): Ein[A] = Ein.Contract(this, right)
  def *(right: Ein[A]): Ein[A] = Ein.Contract(this, right)
  def +(right: Ein[A]): Ein[A] = Ein.ElemAdd(this, right)
  def -(right: Ein[A]): Ein[A] = Ein.ElemSub(this, right)
  def elemMul(right: Ein[A]): Ein[A] = Ein.ElemMul(this, right)

  def outputDims: List[Dim] = this match
    case Param(_, dims, _)   => dims
    case Input(_, dims)      => dims
    case Fill(_, dims)       => dims
    case Ones(dims)          => dims
    case Zeros(dims)         => dims
    case Contract(left, right) =>
      val leftDims = left.outputDims
      val rightDims = right.outputDims
      val leftNames = leftDims.map(_.name).toSet
      val rightNames = rightDims.map(_.name).toSet
      val shared = leftNames.intersect(rightNames)
      leftDims.filterNot(d => shared.contains(d.name)) ++
        rightDims.filterNot(d => shared.contains(d.name))
    case ElemAdd(left, right)  => Ein.broadcastDims(left.outputDims, right.outputDims)
    case ElemSub(left, right)  => Ein.broadcastDims(left.outputDims, right.outputDims)
    case ElemMul(left, right)  => Ein.broadcastDims(left.outputDims, right.outputDims)
    case Activate(_, arg)      => arg.outputDims
    case ActivateDeriv(_, arg) => arg.outputDims
    case ReduceSum(arg, over)  =>
      val overSet = over.toSet
      arg.outputDims.filterNot(d => overSet.contains(d.name))
    case Broadcast(_, targetDims) => targetDims
    case Transpose(arg, perm) =>
      val dimMap = arg.outputDims.map(d => d.name -> d).toMap
      perm.map(name => dimMap(name))
    case Softmax(arg, _)    => arg.outputDims
    case LogSoftmax(arg, _) => arg.outputDims
    case LayerNorm(arg, _, _, _, _) => arg.outputDims
    case Reshape(_, targetDims) => targetDims
    case Slice(arg, dim, from, to) =>
      arg.outputDims.map(d => if d.name == dim then Dim(d.name, to - from) else d)
    case Gather(table, indices, lookupDim) =>
      val tableDims = table.outputDims
      val lookupIdx = tableDims.indexWhere(_.name == lookupDim)
      tableDims.take(lookupIdx) ++ indices.dims ++ tableDims.drop(lookupIdx + 1)
    case Scatter(_, _, _, tableDims) => tableDims

object Ein:
  def broadcastDims(left: List[Dim], right: List[Dim]): List[Dim] =
    val rightMap = right.map(d => d.name -> d).toMap
    val leftNames = left.map(_.name).toSet
    left ++ right.filterNot(d => leftNames.contains(d.name))
