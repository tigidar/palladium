package palladium.ein

import palladium.NumberLike
import scala.reflect.ClassTag

object EinGrad:

  def backward[A: NumberLike: ClassTag](
      expr: Ein[A],
      feed: Map[String, TensorData[A]] = Map.empty[String, TensorData[A]]
  ): Map[String, TensorData[A]] =
    val num = summon[NumberLike[A]]
    // Start with upstream = ones tensor matching output shape
    val outDims = resolveOutputDims(expr, feed)
    val upstream = TensorData.fill(outDims, num.fromInt(1))
    val accum = scala.collection.mutable.Map[String, TensorData[A]]()
    backwardAccum(expr, upstream, feed, accum)
    accum.toMap

  private def resolveOutputDims[A: NumberLike: ClassTag](
      expr: Ein[A],
      feed: Map[String, TensorData[A]]
  ): List[Dim] =
    expr match
      case Ein.Input(id, _) =>
        feed.get(id).map(_.dims).getOrElse(expr.outputDims)
      case _ => expr.outputDims

  private def backwardAccum[A: NumberLike: ClassTag](
      expr: Ein[A],
      upstream: TensorData[A],
      feed: Map[String, TensorData[A]],
      accum: scala.collection.mutable.Map[String, TensorData[A]]
  ): Unit =
    val num = summon[NumberLike[A]]
    expr match
      case Ein.Param(id, _, _) =>
        accum.get(id) match
          case Some(existing) =>
            accum(id) = addTensors(existing, upstream)
          case None =>
            accum(id) = upstream

      case Ein.Input(_, _) => () // Inputs don't have gradients

      case Ein.Fill(_, _) => () // Constants don't have gradients
      case Ein.Ones(_)     => ()
      case Ein.Zeros(_)    => ()

      case Ein.Contract(left, right) =>
        val leftVal = EinEval.eval(left, feed)
        val rightVal = EinEval.eval(right, feed)
        val leftDims = resolveOutputDims(left, feed)
        val rightDims = resolveOutputDims(right, feed)
        // dL/dLeft = Contract(upstream, right) — contracts over shared dims between upstream and right
        val leftGrad = reorderDims(EinEval.evalContract(upstream, rightVal), leftDims)
        // dL/dRight = Contract(upstream, left) — contracts over shared dims between upstream and left
        val rightGrad = reorderDims(EinEval.evalContract(upstream, leftVal), rightDims)
        backwardAccum(left, leftGrad, feed, accum)
        backwardAccum(right, rightGrad, feed, accum)

      case Ein.ElemAdd(left, right) =>
        val leftDims = resolveOutputDims(left, feed)
        val rightDims = resolveOutputDims(right, feed)
        val leftGrad = reduceToDims(upstream, leftDims)
        val rightGrad = reduceToDims(upstream, rightDims)
        backwardAccum(left, leftGrad, feed, accum)
        backwardAccum(right, rightGrad, feed, accum)

      case Ein.ElemSub(left, right) =>
        val leftDims = resolveOutputDims(left, feed)
        val rightDims = resolveOutputDims(right, feed)
        val leftGrad = reduceToDims(upstream, leftDims)
        val rightGrad = negateTensor(reduceToDims(upstream, rightDims))
        backwardAccum(left, leftGrad, feed, accum)
        backwardAccum(right, rightGrad, feed, accum)

      case Ein.ElemMul(left, right) =>
        val leftVal = EinEval.eval(left, feed)
        val rightVal = EinEval.eval(right, feed)
        val leftDims = resolveOutputDims(left, feed)
        val rightDims = resolveOutputDims(right, feed)
        // dL/dLeft = reduceToDims(upstream * rightVal, leftDims)
        val leftGrad = reduceToDims(
          EinEval.evalElemBinary(upstream, rightVal, num.times),
          leftDims
        )
        // dL/dRight = reduceToDims(upstream * leftVal, rightDims)
        val rightGrad = reduceToDims(
          EinEval.evalElemBinary(upstream, leftVal, num.times),
          rightDims
        )
        backwardAccum(left, leftGrad, feed, accum)
        backwardAccum(right, rightGrad, feed, accum)

      case Ein.Activate(f, arg) =>
        // dL/dArg = upstream * f'(argVal)
        val argVal = EinEval.eval(arg, feed)
        val derivVal = EinEval.mapTensor(argVal)(x => Activation.deriv(f, x))
        val argGrad = EinEval.evalElemBinary(upstream, derivVal, num.times)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.ActivateDeriv(_, _) => () // Leaf in symbolic grad expressions

      case Ein.ReduceSum(arg, over) =>
        // Gradient of sum is broadcast
        val argDims = resolveOutputDims(arg, feed)
        val argGrad = EinEval.evalBroadcast(upstream, argDims)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.Broadcast(arg, targetDims) =>
        // Gradient of broadcast is reduce
        val argDims = resolveOutputDims(arg, feed)
        val broadcastedNames = targetDims.map(_.name).toSet -- argDims.map(_.name).toSet
        val argGrad = EinEval.evalReduceSum(upstream, broadcastedNames.toList)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.Transpose(arg, perm) =>
        // Inverse permutation
        val argDims = resolveOutputDims(arg, feed)
        val argNames = argDims.map(_.name)
        val inversePerm = argNames // original order
        val argGrad = EinEval.evalTranspose(upstream, inversePerm)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.Softmax(arg, overDim) =>
        // grad_i = s_i * (upstream_i - sum_j(upstream_j * s_j))
        val argVal = EinEval.eval(arg, feed)
        val softmaxVal = EinEval.evalSoftmax(argVal, overDim)
        val dot = EinEval.evalElemBinary(upstream, softmaxVal, num.times)
        val sumDot = EinEval.evalBroadcast(
          EinEval.evalReduceSum(dot, List(overDim)),
          argVal.dims
        )
        val diff = EinEval.evalElemBinary(upstream, sumDot, num.minus)
        val argGrad = EinEval.evalElemBinary(softmaxVal, diff, num.times)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.LogSoftmax(arg, overDim) =>
        // grad_i = upstream_i - softmax_i * sum_j(upstream_j)
        val argVal = EinEval.eval(arg, feed)
        val softmaxVal = EinEval.evalSoftmax(argVal, overDim)
        val sumUpstream = EinEval.evalBroadcast(
          EinEval.evalReduceSum(upstream, List(overDim)),
          argVal.dims
        )
        val scaled = EinEval.evalElemBinary(softmaxVal, sumUpstream, num.times)
        val argGrad = EinEval.evalElemBinary(upstream, scaled, num.minus)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.LayerNorm(arg, scale, bias, overDims, eps) =>
        // Forward values needed for chain rule
        val argVal = EinEval.eval(arg, feed)
        val scaleVal = EinEval.eval(scale, feed)
        val one = num.fromInt(1)
        val epsA = num.div(one, num.fromInt((1.0 / eps).toInt))
        val count = overDims.map(dim => argVal.dims.find(_.name == dim).get.size).product
        val countA = num.fromInt(count)

        // Recompute forward internals
        val sum = EinEval.evalReduceSum(argVal, overDims)
        val mean = EinEval.mapTensor(sum)(x => num.div(x, countA))
        val meanBcast = EinEval.evalBroadcast(mean, argVal.dims)
        val centered = EinEval.evalElemBinary(argVal, meanBcast, num.minus)
        val squared = EinEval.mapTensor(centered)(x => num.times(x, x))
        val varSum = EinEval.evalReduceSum(squared, overDims)
        val variance = EinEval.mapTensor(varSum)(x => num.div(x, countA))
        val varBcast = EinEval.evalBroadcast(variance, argVal.dims)
        val invStd = EinEval.mapTensor(varBcast)(v => num.div(one, num.sqrt(num.plus(v, epsA))))
        val normalized = EinEval.evalElemBinary(centered, invStd, num.times)

        // Gradient for bias: dL/dbias = sum(upstream) over broadcast dims
        val biasDims = resolveOutputDims(bias, feed)
        val biasGrad = reduceToDims(upstream, biasDims)
        backwardAccum(bias, biasGrad, feed, accum)

        // Gradient for scale: dL/dscale = sum(upstream * normalized) over broadcast dims
        val scaleDims = resolveOutputDims(scale, feed)
        val scaleGrad = reduceToDims(
          EinEval.evalElemBinary(upstream, normalized, num.times),
          scaleDims
        )
        backwardAccum(scale, scaleGrad, feed, accum)

        // Gradient for arg: chain rule through normalization
        // dL/dx = (1/N) * invStd * (N * dy_scaled - sum(dy_scaled) - normalized * sum(dy_scaled * normalized))
        // where dy_scaled = upstream * scale
        val scaleBcast = EinEval.evalBroadcast(scaleVal, argVal.dims)
        val dyScaled = EinEval.evalElemBinary(upstream, scaleBcast, num.times)

        val sumDy = EinEval.evalBroadcast(
          EinEval.evalReduceSum(dyScaled, overDims),
          argVal.dims
        )
        val dyNorm = EinEval.evalElemBinary(dyScaled, normalized, num.times)
        val sumDyNorm = EinEval.evalBroadcast(
          EinEval.evalReduceSum(dyNorm, overDims),
          argVal.dims
        )
        // N * dy_scaled - sumDy - normalized * sumDyNorm
        val nDy = EinEval.mapTensor(dyScaled)(x => num.times(countA, x))
        val term1 = EinEval.evalElemBinary(nDy, sumDy, num.minus)
        val normTimesSum = EinEval.evalElemBinary(normalized, sumDyNorm, num.times)
        val numerator = EinEval.evalElemBinary(term1, normTimesSum, num.minus)
        // Multiply by invStd / N
        val invStdOverN = EinEval.mapTensor(invStd)(x => num.div(x, countA))
        val argGrad = EinEval.evalElemBinary(numerator, invStdOverN, num.times)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.Reshape(arg, targetDims) =>
        val argDims = resolveOutputDims(arg, feed)
        val argGrad = TensorData(upstream.data.clone(), argDims, TensorData.computeStrides(argDims))
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.Slice(arg, dim, from, to) =>
        // Gradient of slice is pad: embed upstream into zeros at the sliced positions
        val argDims = resolveOutputDims(arg, feed)
        val argGrad = EinEval.evalSlicePad(upstream, argDims, dim, from)
        backwardAccum(arg, argGrad, feed, accum)

      case Ein.Gather(table, indices, lookupDim) =>
        // Gradient of gather is scatter: distribute upstream gradient back to table positions
        val tableDims = resolveOutputDims(table, feed)
        val tableGrad = EinEval.evalScatter(upstream, indices, lookupDim, tableDims)
        backwardAccum(table, tableGrad, feed, accum)

      case Ein.Scatter(src, indices, lookupDim, tableDims) =>
        // Gradient of scatter is gather: pick gradient from upstream at the indexed positions
        val srcGrad = EinEval.evalGather(upstream, indices, lookupDim)
        backwardAccum(src, srcGrad, feed, accum)

  private def addTensors[A: NumberLike: ClassTag](
      a: TensorData[A],
      b: TensorData[A]
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    EinEval.evalElemBinary(a, b, num.plus)

  private def negateTensor[A: NumberLike: ClassTag](
      t: TensorData[A]
  ): TensorData[A] =
    val num = summon[NumberLike[A]]
    EinEval.mapTensor(t)(num.negate)

  private def reorderDims[A: NumberLike: ClassTag](
      tensor: TensorData[A],
      targetDims: List[Dim]
  ): TensorData[A] =
    val targetNames = targetDims.map(_.name)
    val currentNames = tensor.dims.map(_.name)
    if currentNames == targetNames then tensor
    else EinEval.evalTranspose(tensor, targetNames)

  private def reduceToDims[A: NumberLike: ClassTag](
      tensor: TensorData[A],
      targetDims: List[Dim]
  ): TensorData[A] =
    val targetNames = targetDims.map(_.name).toSet
    val toReduce = tensor.dims.map(_.name).filterNot(targetNames.contains)
    if toReduce.isEmpty then tensor
    else EinEval.evalReduceSum(tensor, toReduce)

  object syntax:
    extension [A: NumberLike: ClassTag](expr: Ein[A])
      def backward(feed: Map[String, TensorData[A]] = Map.empty[String, TensorData[A]]): Map[String, TensorData[A]] =
        EinGrad.backward(expr, feed)
