package palladium.ein

import palladium.NumberLike
import scala.reflect.ClassTag

object EinEval:

  def eval[A: NumberLike: ClassTag](
      expr: Ein[A],
      feed: Map[String, TensorData[A]] = Map.empty[String, TensorData[A]]
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    expr match
      case Ein.Param(_, _, data) => data
      case Ein.Input(id, dims) =>
        feed.getOrElse(id, throw RuntimeException(s"Missing feed for input '$id'"))
      case Ein.Fill(value, dims) => TensorData.fill(dims, value)
      case Ein.Ones(dims) => TensorData.fill(dims, num.fromInt(1))
      case Ein.Zeros(dims) => TensorData.zeros[A](dims)

      case Ein.Contract(left, right) =>
        evalContract(eval(left, feed), eval(right, feed))

      case Ein.ElemAdd(left, right) =>
        evalElemBinary(eval(left, feed), eval(right, feed), num.plus)
      case Ein.ElemSub(left, right) =>
        evalElemBinary(eval(left, feed), eval(right, feed), num.minus)
      case Ein.ElemMul(left, right) =>
        evalElemBinary(eval(left, feed), eval(right, feed), num.times)

      case Ein.Activate(f, arg) =>
        val argVal = eval(arg, feed)
        mapTensor(argVal)(x => Activation(f, x))
      case Ein.ActivateDeriv(f, arg) =>
        val argVal = eval(arg, feed)
        mapTensor(argVal)(x => Activation.deriv(f, x))

      case Ein.ReduceSum(arg, over) =>
        evalReduceSum(eval(arg, feed), over)
      case Ein.Broadcast(arg, targetDims) =>
        evalBroadcast(eval(arg, feed), targetDims)
      case Ein.Transpose(arg, perm) =>
        evalTranspose(eval(arg, feed), perm)
      case Ein.Softmax(arg, overDim) =>
        evalSoftmax(eval(arg, feed), overDim)
      case Ein.LogSoftmax(arg, overDim) =>
        evalLogSoftmax(eval(arg, feed), overDim)
      case Ein.LayerNorm(arg, scale, bias, overDims, eps) =>
        evalLayerNorm(eval(arg, feed), eval(scale, feed), eval(bias, feed), overDims, eps)

      case Ein.Reshape(arg, targetDims) =>
        val argVal = eval(arg, feed)
        TensorData(argVal.data.clone(), targetDims, TensorData.computeStrides(targetDims))

      case Ein.Slice(arg, dim, from, to) =>
        evalSlice(eval(arg, feed), dim, from, to)

      case Ein.Gather(table, indices, lookupDim) =>
        evalGather(eval(table, feed), indices, lookupDim)
      case Ein.Scatter(src, indices, lookupDim, tableDims) =>
        evalScatter(eval(src, feed), indices, lookupDim, tableDims)

  private[ein] def mapTensor[A: NumberLike: ClassTag](t: TensorData[A])(f: A => A): TensorData[A] =
    val result = new Array[A](t.data.length)
    var i = 0
    while i < t.data.length do
      result(i) = f(t.data(i))
      i += 1
    TensorData(result, t.dims, t.strides)

  def evalContract[A: NumberLike: ClassTag](
      left: TensorData[A],
      right: TensorData[A]
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    val leftDims = left.dims
    val rightDims = right.dims
    val leftNames = leftDims.map(_.name).toSet
    val rightNames = rightDims.map(_.name).toSet
    val sharedNames = leftNames.intersect(rightNames)

    val leftFreeDims = leftDims.filterNot(d => sharedNames.contains(d.name))
    val rightFreeDims = rightDims.filterNot(d => sharedNames.contains(d.name))
    val sharedDims = leftDims.filter(d => sharedNames.contains(d.name))
    val outDims = leftFreeDims ++ rightFreeDims

    val output = TensorData.zeros[A](outDims)
    if output.data.isEmpty then return output

    // Build all dims for iteration: leftFree ++ rightFree ++ shared
    val allDims = outDims ++ sharedDims
    val allSizes = allDims.map(_.size).toArray
    val ndim = allDims.length
    if ndim == 0 then
      // scalar * scalar
      output.data(0) = num.times(left.data(0), right.data(0))
      return output

    // Precompute index mappings: for each allDims position, what's its index in left/right dims?
    val leftDimIndex = leftDims.map(_.name).zipWithIndex.toMap
    val rightDimIndex = rightDims.map(_.name).zipWithIndex.toMap
    val allNames = allDims.map(_.name)

    val leftMapping = allNames.map(n => leftDimIndex.getOrElse(n, -1)).toArray
    val rightMapping = allNames.map(n => rightDimIndex.getOrElse(n, -1)).toArray

    // Number of output dims (leftFree + rightFree)
    val outNdim = outDims.length
    val outStrides = output.strides.toArray
    val leftStrides = left.strides.toArray
    val rightStrides = right.strides.toArray

    // Iterate over all combinations using mixed-radix counter
    val indices = new Array[Int](ndim)
    val totalIter = allSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      // Compute offsets
      var leftOff = 0
      var rightOff = 0
      var outOff = 0
      var d = 0
      while d < ndim do
        val idx = indices(d)
        if leftMapping(d) >= 0 then leftOff += idx * leftStrides(leftMapping(d))
        if rightMapping(d) >= 0 then rightOff += idx * rightStrides(rightMapping(d))
        if d < outNdim then outOff += idx * outStrides(d)
        d += 1

      output.data(outOff) = num.plus(
        output.data(outOff),
        num.times(left.data(leftOff), right.data(rightOff))
      )

      // Increment mixed-radix counter (least significant last)
      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        indices(k) += 1
        if indices(k) >= allSizes(k) then
          indices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  def evalElemBinary[A: NumberLike: ClassTag](
      left: TensorData[A],
      right: TensorData[A],
      op: (A, A) => A
  ): TensorData[A] =
    val outDims = Ein.broadcastDims(left.dims, right.dims)
    val output = TensorData.zeros[A](outDims)
    if output.data.isEmpty then return output

    val allSizes = outDims.map(_.size).toArray
    val ndim = outDims.length
    if ndim == 0 then
      output.data(0) = op(left.data(0), right.data(0))
      return output

    // Precompute projection: for each output dim, what's the index in left/right?
    val leftDimIndex = left.dims.map(_.name).zipWithIndex.toMap
    val rightDimIndex = right.dims.map(_.name).zipWithIndex.toMap
    val outNames = outDims.map(_.name)

    val leftMapping = outNames.map(n => leftDimIndex.getOrElse(n, -1)).toArray
    val rightMapping = outNames.map(n => rightDimIndex.getOrElse(n, -1)).toArray
    val leftStrides = left.strides.toArray
    val rightStrides = right.strides.toArray
    val outStrides = output.strides.toArray

    val indices = new Array[Int](ndim)
    val totalIter = allSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var leftOff = 0
      var rightOff = 0
      var outOff = 0
      var d = 0
      while d < ndim do
        val idx = indices(d)
        if leftMapping(d) >= 0 then leftOff += idx * leftStrides(leftMapping(d))
        if rightMapping(d) >= 0 then rightOff += idx * rightStrides(rightMapping(d))
        outOff += idx * outStrides(d)
        d += 1

      output.data(outOff) = op(left.data(leftOff), right.data(rightOff))

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        indices(k) += 1
        if indices(k) >= allSizes(k) then
          indices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  def evalReduceSum[A: NumberLike: ClassTag](
      input: TensorData[A],
      over: List[String]
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    val overSet = over.toSet
    val outDims = input.dims.filterNot(d => overSet.contains(d.name))
    val output = TensorData.zeros[A](outDims)

    val inDims = input.dims
    val inSizes = inDims.map(_.size).toArray
    val ndim = inDims.length
    if ndim == 0 then
      output.data(0) = input.data(0)
      return output

    // Map each input dim to output dim index (-1 if being reduced)
    val outDimIndex = outDims.map(_.name).zipWithIndex.toMap
    val inputToOut = inDims.map(d => outDimIndex.getOrElse(d.name, -1)).toArray
    val inStrides = input.strides.toArray
    val outStrides = output.strides.toArray

    val indices = new Array[Int](ndim)
    val totalIter = inSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var inOff = 0
      var outOff = 0
      var d = 0
      while d < ndim do
        val idx = indices(d)
        inOff += idx * inStrides(d)
        if inputToOut(d) >= 0 then outOff += idx * outStrides(inputToOut(d))
        d += 1

      output.data(outOff) = num.plus(output.data(outOff), input.data(inOff))

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        indices(k) += 1
        if indices(k) >= inSizes(k) then
          indices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  def evalBroadcast[A: NumberLike: ClassTag](
      input: TensorData[A],
      targetDims: List[Dim]
  ): TensorData[A] =
    val outDims = targetDims
    val output = TensorData.zeros[A](outDims)
    val outSizes = outDims.map(_.size).toArray
    val ndim = outDims.length
    if ndim == 0 then
      output.data(0) = input.data(0)
      return output

    val inDimIndex = input.dims.map(_.name).zipWithIndex.toMap
    val outNames = outDims.map(_.name)
    val inMapping = outNames.map(n => inDimIndex.getOrElse(n, -1)).toArray
    val inStrides = input.strides.toArray
    val outStrides = output.strides.toArray

    val indices = new Array[Int](ndim)
    val totalIter = outSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var inOff = 0
      var outOff = 0
      var d = 0
      while d < ndim do
        val idx = indices(d)
        if inMapping(d) >= 0 then inOff += idx * inStrides(inMapping(d))
        outOff += idx * outStrides(d)
        d += 1

      output.data(outOff) = input.data(inOff)

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        indices(k) += 1
        if indices(k) >= outSizes(k) then
          indices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  def evalTranspose[A: NumberLike: ClassTag](
      input: TensorData[A],
      perm: List[String]
  ): TensorData[A] =
    val inDimIndex = input.dims.map(_.name).zipWithIndex.toMap
    val permIndices = perm.map(n => inDimIndex(n))
    val outDims = permIndices.map(i => input.dims(i))
    val output = TensorData.zeros[A](outDims)

    val outSizes = outDims.map(_.size).toArray
    val ndim = outDims.length
    if ndim == 0 then
      output.data(0) = input.data(0)
      return output

    val inStrides = input.strides.toArray
    val outStrides = output.strides.toArray

    val indices = new Array[Int](ndim)
    val totalIter = outSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var inOff = 0
      var outOff = 0
      var d = 0
      while d < ndim do
        val idx = indices(d)
        inOff += idx * inStrides(permIndices(d))
        outOff += idx * outStrides(d)
        d += 1

      output.data(outOff) = input.data(inOff)

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        indices(k) += 1
        if indices(k) >= outSizes(k) then
          indices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  def evalReduceMax[A: NumberLike: ClassTag](
      input: TensorData[A],
      over: List[String]
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    val overSet = over.toSet
    val outDims = input.dims.filterNot(d => overSet.contains(d.name))
    val output = TensorData.fill(outDims, num.negativeInfinity)

    val inDims = input.dims
    val inSizes = inDims.map(_.size).toArray
    val ndim = inDims.length
    if ndim == 0 then
      output.data(0) = input.data(0)
      return output

    val outDimIndex = outDims.map(_.name).zipWithIndex.toMap
    val inputToOut = inDims.map(d => outDimIndex.getOrElse(d.name, -1)).toArray
    val inStrides = input.strides.toArray
    val outStrides = output.strides.toArray

    val indices = new Array[Int](ndim)
    val totalIter = inSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var inOff = 0
      var outOff = 0
      var d = 0
      while d < ndim do
        val idx = indices(d)
        inOff += idx * inStrides(d)
        if inputToOut(d) >= 0 then outOff += idx * outStrides(inputToOut(d))
        d += 1

      output.data(outOff) = num.max(output.data(outOff), input.data(inOff))

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        indices(k) += 1
        if indices(k) >= inSizes(k) then
          indices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  private[ein] def evalSoftmax[A: NumberLike: ClassTag](
      input: TensorData[A],
      overDim: String
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    // max-subtraction trick for numerical stability
    val maxVal = evalReduceMax(input, List(overDim))
    val shifted = evalElemBinary(input, evalBroadcast(maxVal, input.dims), num.minus)
    val exped = mapTensor(shifted)(num.exp)
    val sumExp = evalReduceSum(exped, List(overDim))
    evalElemBinary(exped, evalBroadcast(sumExp, input.dims), num.div)

  private[ein] def evalLogSoftmax[A: NumberLike: ClassTag](
      input: TensorData[A],
      overDim: String
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    // log-sum-exp trick for numerical stability
    val maxVal = evalReduceMax(input, List(overDim))
    val shifted = evalElemBinary(input, evalBroadcast(maxVal, input.dims), num.minus)
    val exped = mapTensor(shifted)(num.exp)
    val sumExp = evalReduceSum(exped, List(overDim))
    val logSumExp = mapTensor(sumExp)(num.log)
    evalElemBinary(shifted, evalBroadcast(logSumExp, input.dims), num.minus)

  private[ein] def evalLayerNorm[A: NumberLike: ClassTag](
      input: TensorData[A],
      scale: TensorData[A],
      bias: TensorData[A],
      overDims: List[String],
      eps: Double
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    val epsA = num.div(num.fromInt(1), num.fromInt((1.0 / eps).toInt)) // approximate eps from Double
    val count = overDims.map(dim => input.dims.find(_.name == dim).get.size).product
    val countA = num.fromInt(count)

    // mean = sum(x, overDims) / count
    val sum = evalReduceSum(input, overDims)
    val mean = mapTensor(sum)(x => num.div(x, countA))
    val meanBcast = evalBroadcast(mean, input.dims)

    // centered = x - mean
    val centered = evalElemBinary(input, meanBcast, num.minus)

    // variance = sum((x - mean)^2, overDims) / count
    val squared = mapTensor(centered)(x => num.times(x, x))
    val varSum = evalReduceSum(squared, overDims)
    val variance = mapTensor(varSum)(x => num.div(x, countA))
    val varBcast = evalBroadcast(variance, input.dims)

    // normalized = centered / sqrt(variance + eps)
    val invStd = mapTensor(varBcast)(v => num.div(num.fromInt(1), num.sqrt(num.plus(v, epsA))))
    val normalized = evalElemBinary(centered, invStd, num.times)

    // output = scale * normalized + bias (with broadcasting)
    val scaleBcast = evalBroadcast(scale, input.dims)
    val biasBcast = evalBroadcast(bias, input.dims)
    val scaled = evalElemBinary(normalized, scaleBcast, num.times)
    evalElemBinary(scaled, biasBcast, num.plus)

  private[ein] def evalGather[A: NumberLike: ClassTag](
      table: TensorData[A],
      indices: TensorData[Int],
      lookupDim: String
  ): TensorData[A] =
    val tableDims = table.dims
    val lookupIdx = tableDims.indexWhere(_.name == lookupDim)
    val beforeDims = tableDims.take(lookupIdx)
    val afterDims = tableDims.drop(lookupIdx + 1)
    val outDims = beforeDims ++ indices.dims ++ afterDims
    val output = TensorData.zeros[A](outDims)
    if output.data.isEmpty then return output

    val outSizes = outDims.map(_.size).toArray
    val ndim = outDims.length
    val outStrides = output.strides.toArray
    val tableStrides = table.strides.toArray
    val indexStrides = indices.strides.toArray

    val beforeCount = beforeDims.length
    val indexCount = indices.dims.length
    val afterCount = afterDims.length

    val outIndices = new Array[Int](ndim)
    val totalIter = outSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      // Compute output offset
      var outOff = 0
      var d = 0
      while d < ndim do
        outOff += outIndices(d) * outStrides(d)
        d += 1

      // Compute index into indices tensor
      var idxOff = 0
      d = 0
      while d < indexCount do
        idxOff += outIndices(beforeCount + d) * indexStrides(d)
        d += 1
      val lookupVal = indices.data(idxOff)

      // Compute table offset: before dims + lookup value + after dims
      var tableOff = 0
      d = 0
      while d < beforeCount do
        tableOff += outIndices(d) * tableStrides(d)
        d += 1
      tableOff += lookupVal * tableStrides(lookupIdx)
      d = 0
      while d < afterCount do
        tableOff += outIndices(beforeCount + indexCount + d) * tableStrides(lookupIdx + 1 + d)
        d += 1

      output.data(outOff) = table.data(tableOff)

      // Increment mixed-radix counter
      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        outIndices(k) += 1
        if outIndices(k) >= outSizes(k) then
          outIndices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  private[ein] def evalScatter[A: NumberLike: ClassTag](
      src: TensorData[A],
      indices: TensorData[Int],
      lookupDim: String,
      tableDims: List[Dim]
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    val output = TensorData.zeros[A](tableDims)
    if output.data.isEmpty then return output

    val srcDims = src.dims
    val lookupIdx = tableDims.indexWhere(_.name == lookupDim)
    val beforeDims = tableDims.take(lookupIdx)
    val afterDims = tableDims.drop(lookupIdx + 1)
    val beforeCount = beforeDims.length
    val indexCount = indices.dims.length
    val afterCount = afterDims.length

    val srcSizes = srcDims.map(_.size).toArray
    val ndim = srcDims.length
    val srcStrides = src.strides.toArray
    val tableStrides = output.strides.toArray
    val indexStrides = indices.strides.toArray

    val srcIndices = new Array[Int](ndim)
    val totalIter = srcSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var srcOff = 0
      var d = 0
      while d < ndim do
        srcOff += srcIndices(d) * srcStrides(d)
        d += 1

      // Compute index into indices tensor
      var idxOff = 0
      d = 0
      while d < indexCount do
        idxOff += srcIndices(beforeCount + d) * indexStrides(d)
        d += 1
      val lookupVal = indices.data(idxOff)

      // Compute table offset
      var tableOff = 0
      d = 0
      while d < beforeCount do
        tableOff += srcIndices(d) * tableStrides(d)
        d += 1
      tableOff += lookupVal * tableStrides(lookupIdx)
      d = 0
      while d < afterCount do
        tableOff += srcIndices(beforeCount + indexCount + d) * tableStrides(lookupIdx + 1 + d)
        d += 1

      output.data(tableOff) = num.plus(output.data(tableOff), src.data(srcOff))

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        srcIndices(k) += 1
        if srcIndices(k) >= srcSizes(k) then
          srcIndices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  private[ein] def evalSlice[A: NumberLike: ClassTag](
      input: TensorData[A],
      dim: String,
      from: Int,
      to: Int
  ): TensorData[A] =
    val sliceDimIdx = input.dims.indexWhere(_.name == dim)
    val outDims = input.dims.updated(sliceDimIdx, Dim(dim, to - from))
    val output = TensorData.zeros[A](outDims)
    if output.data.isEmpty then return output

    val outSizes = outDims.map(_.size).toArray
    val ndim = outDims.length
    val outStrides = output.strides.toArray
    val inStrides = input.strides.toArray

    val indices = new Array[Int](ndim)
    val totalIter = outSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var outOff = 0
      var inOff = 0
      var d = 0
      while d < ndim do
        val idx = indices(d)
        outOff += idx * outStrides(d)
        if d == sliceDimIdx then
          inOff += (idx + from) * inStrides(d)
        else
          inOff += idx * inStrides(d)
        d += 1

      output.data(outOff) = input.data(inOff)

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        indices(k) += 1
        if indices(k) >= outSizes(k) then
          indices(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output

  private[ein] def evalSlicePad[A: NumberLike: ClassTag](
      src: TensorData[A],
      targetDims: List[Dim],
      dim: String,
      offset: Int
  ): TensorData[A] =
    val output = TensorData.zeros[A](targetDims)
    if src.data.isEmpty then return output

    val sliceDimIdx = targetDims.indexWhere(_.name == dim)
    val srcSizes = src.dims.map(_.size).toArray
    val ndim = src.dims.length
    val srcStrides = src.strides.toArray
    val outStrides = output.strides.toArray

    val srcIdx = new Array[Int](ndim)
    val totalIter = srcSizes.foldLeft(1)(_ * _)
    var iter = 0
    while iter < totalIter do
      var srcOff = 0
      var outOff = 0
      var d = 0
      while d < ndim do
        val idx = srcIdx(d)
        srcOff += idx * srcStrides(d)
        if d == sliceDimIdx then
          outOff += (idx + offset) * outStrides(d)
        else
          outOff += idx * outStrides(d)
        d += 1

      output.data(outOff) = src.data(srcOff)

      var carry = true
      var k = ndim - 1
      while k >= 0 && carry do
        srcIdx(k) += 1
        if srcIdx(k) >= srcSizes(k) then
          srcIdx(k) = 0
        else
          carry = false
        k -= 1
      iter += 1
    output
