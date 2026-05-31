package palladium.codegen

import palladium.ein.*

object Lower:

  private case class LowerState(
    nextId: Int,
    ops: Vector[LowOp],
    params: Vector[(TensorRef, String)],
    inputs: Vector[(TensorRef, String)],
    temps: Vector[TensorRef],
    paramCache: Map[String, TensorRef],
    inputCache: Map[String, TensorRef],
    indicesData: Map[Int, Array[Int]] = Map.empty
  ):
    def alloc(shape: List[Int]): (TensorRef, LowerState) =
      val ref = TensorRef(nextId, shape)
      (ref, copy(nextId = nextId + 1, temps = temps :+ ref))

    def allocParam(name: String, shape: List[Int]): (TensorRef, LowerState) =
      val ref = TensorRef(nextId, shape)
      (ref, copy(nextId = nextId + 1, params = params :+ (ref, name), paramCache = paramCache + (name -> ref)))

    def allocInput(name: String, shape: List[Int]): (TensorRef, LowerState) =
      val ref = TensorRef(nextId, shape)
      (ref, copy(nextId = nextId + 1, inputs = inputs :+ (ref, name), inputCache = inputCache + (name -> ref)))

    def emit(op: LowOp): LowerState = copy(ops = ops :+ op)

    def allocIndices(data: Array[Int]): (Int, LowerState) =
      val id = nextId
      (id, copy(nextId = nextId + 1, indicesData = indicesData + (id -> data)))

  private def initial: LowerState =
    LowerState(0, Vector.empty, Vector.empty, Vector.empty, Vector.empty, Map.empty, Map.empty, Map.empty)

  def lower(expr: Ein[Double]): LowProgram =
    val (outRef, finalState) = lowerExpr(expr, initial)
    LowProgram(
      ops = finalState.ops,
      params = finalState.params,
      inputs = finalState.inputs,
      outputs = Vector(outRef),
      tempBuffers = finalState.temps,
      indicesData = finalState.indicesData
    )

  private def convertActivation(f: Activation): ActivationType = f match
    case Activation.ReLU    => ActivationType.ReLU
    case Activation.Sigmoid => ActivationType.Sigmoid
    case Activation.Tanh    => ActivationType.Tanh
    case Activation.GELU    => ActivationType.GELU
    case Activation.Swish   => ActivationType.Swish

  private def dimsToShape(dims: List[Dim]): List[Int] = dims.map(_.size)

  private def lowerExpr(expr: Ein[Double], st: LowerState): (TensorRef, LowerState) =
    expr match
      case Ein.Param(id, dims, _) =>
        st.paramCache.get(id) match
          case Some(ref) => (ref, st)
          case None      => st.allocParam(id, dimsToShape(dims))

      case Ein.Input(id, dims) =>
        st.inputCache.get(id) match
          case Some(ref) => (ref, st)
          case None      => st.allocInput(id, dimsToShape(dims))

      case Ein.Fill(_, dims) =>
        val shape = dimsToShape(dims)
        val (ref, s1) = st.alloc(shape)
        (ref, s1.emit(LowOp.FillOnes(ref)))

      case Ein.Ones(dims) =>
        val shape = dimsToShape(dims)
        val (ref, s1) = st.alloc(shape)
        (ref, s1.emit(LowOp.FillOnes(ref)))

      case Ein.Zeros(dims) =>
        val shape = dimsToShape(dims)
        val (ref, s1) = st.alloc(shape)
        (ref, s1.emit(LowOp.FillZeros(ref)))

      case Ein.Contract(left, right) =>
        val (leftRef, s1) = lowerExpr(left, st)
        val (rightRef, s2) = lowerExpr(right, s1)
        val leftDims = left.outputDims
        val rightDims = right.outputDims
        val leftNames = leftDims.map(_.name).toSet
        val rightNames = rightDims.map(_.name).toSet
        val shared = leftNames.intersect(rightNames)
        val leftFree = leftDims.filterNot(d => shared.contains(d.name))
        val rightFree = rightDims.filterNot(d => shared.contains(d.name))
        val contractSize = leftDims.filter(d => shared.contains(d.name)).map(_.size).product.max(1)
        val outShape = dimsToShape(leftFree ++ rightFree)
        val (outRef, s3) = s2.alloc(outShape)
        val op = LowOp.MatMul(outRef, leftRef, rightRef, outShape, leftFree.size, rightFree.size, contractSize)
        (outRef, s3.emit(op))

      case Ein.ElemAdd(left, right) =>
        lowerBinary(left, right, BinaryOp.Add, st)

      case Ein.ElemSub(left, right) =>
        lowerBinary(left, right, BinaryOp.Sub, st)

      case Ein.ElemMul(left, right) =>
        lowerBinary(left, right, BinaryOp.Mul, st)

      case Ein.Activate(f, arg) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val outShape = dimsToShape(arg.outputDims)
        val (outRef, s2) = s1.alloc(outShape)
        (outRef, s2.emit(LowOp.Activate(outRef, argRef, convertActivation(f))))

      case Ein.ActivateDeriv(f, arg) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val outShape = dimsToShape(arg.outputDims)
        val (outRef, s2) = s1.alloc(outShape)
        (outRef, s2.emit(LowOp.ActivateDeriv(outRef, argRef, convertActivation(f))))

      case Ein.ReduceSum(arg, over) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val argDims = arg.outputDims
        val overSet = over.toSet
        val keepDimSizes = argDims.filterNot(d => overSet.contains(d.name)).map(_.size)
        val reduceDimSizes = argDims.filter(d => overSet.contains(d.name)).map(_.size)
        val outShape = keepDimSizes
        val (outRef, s2) = s1.alloc(if outShape.isEmpty then List(1) else outShape)
        (outRef, s2.emit(LowOp.ReduceSum(outRef, argRef, keepDimSizes, reduceDimSizes)))

      case Ein.Broadcast(arg, targetDims) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val targetShape = dimsToShape(targetDims)
        val (outRef, s2) = s1.alloc(targetShape)
        (outRef, s2.emit(LowOp.Broadcast(outRef, argRef, targetShape)))

      case Ein.Transpose(arg, perm) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val argDims = arg.outputDims
        val dimMap = argDims.map(d => d.name -> d).toMap
        val outDims = perm.map(name => dimMap(name))
        val outShape = dimsToShape(outDims)
        val (outRef, s2) = s1.alloc(outShape)
        (outRef, s2.emit(LowOp.Copy(outRef, argRef)))

      case Ein.Softmax(arg, overDim) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val argDims = arg.outputDims
        val dimSize = argDims.find(_.name == overDim).map(_.size).getOrElse(1)
        val outShape = dimsToShape(argDims)
        val (outRef, s2) = s1.alloc(outShape)
        (outRef, s2.emit(LowOp.Softmax(outRef, argRef, dimSize)))

      case Ein.LogSoftmax(arg, overDim) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val argDims = arg.outputDims
        val dimSize = argDims.find(_.name == overDim).map(_.size).getOrElse(1)
        val outShape = dimsToShape(argDims)
        val (outRef, s2) = s1.alloc(outShape)
        (outRef, s2.emit(LowOp.LogSoftmax(outRef, argRef, dimSize)))

      case Ein.LayerNorm(arg, scale, bias, overDims, eps) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val (scaleRef, s2) = lowerExpr(scale, s1)
        val (biasRef, s3) = lowerExpr(bias, s2)
        val argDims = arg.outputDims
        val overSet = overDims.toSet
        val normSize = argDims.filter(d => overSet.contains(d.name)).map(_.size).product.max(1)
        val outShape = dimsToShape(argDims)
        val (outRef, s4) = s3.alloc(outShape)
        (outRef, s4.emit(LowOp.LayerNorm(outRef, argRef, scaleRef, biasRef, normSize, eps)))

      case Ein.Reshape(arg, targetDims) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val targetShape = dimsToShape(targetDims)
        val (outRef, s2) = s1.alloc(targetShape)
        (outRef, s2.emit(LowOp.Reshape(outRef, argRef, targetShape)))

      case Ein.Slice(arg, dim, from, to) =>
        val (argRef, s1) = lowerExpr(arg, st)
        val outDims = expr.outputDims
        val outShape = dimsToShape(outDims)
        val (outRef, s2) = s1.alloc(outShape)
        (outRef, s2.emit(LowOp.Copy(outRef, argRef)))

      case Ein.Gather(table, indices, lookupDim) =>
        val (tableRef, s1) = lowerExpr(table, st)
        val outDims = expr.outputDims
        val outShape = dimsToShape(outDims)
        val indicesSize = indices.dims.map(_.size).product.max(1)
        val (indicesId, s2) = s1.allocIndices(indices.data.map(_.toInt))
        val (outRef, s3) = s2.alloc(outShape)
        (outRef, s3.emit(LowOp.Gather(outRef, tableRef, indicesId, indicesSize)))

      case Ein.Scatter(src, indices, lookupDim, tableDims) =>
        val (srcRef, s1) = lowerExpr(src, st)
        val outShape = dimsToShape(tableDims)
        val indicesSize = indices.dims.map(_.size).product.max(1)
        val (indicesId, s2) = s1.allocIndices(indices.data.map(_.toInt))
        val (outRef, s3) = s2.alloc(outShape)
        (outRef, s3.emit(LowOp.Scatter(outRef, srcRef, indicesId, indicesSize)))

  private def lowerBinary(left: Ein[Double], right: Ein[Double], op: BinaryOp, st: LowerState): (TensorRef, LowerState) =
    val (leftRef, s1) = lowerExpr(left, st)
    val (rightRef, s2) = lowerExpr(right, s1)
    val outDims = Ein.broadcastDims(left.outputDims, right.outputDims)
    val outShape = dimsToShape(outDims)
    val (outRef, s3) = s2.alloc(outShape)
    (outRef, s3.emit(LowOp.ElemBinary(outRef, leftRef, rightRef, op)))
