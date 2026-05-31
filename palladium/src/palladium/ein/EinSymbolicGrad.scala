package palladium.ein

object EinSymbolicGrad:

  def backward[A](expr: Ein[A]): Map[String, Ein[A]] =
    val accum = scala.collection.mutable.Map[String, Ein[A]]()
    backwardAccum(expr, None, accum)
    accum.toMap

  private def backwardAccum[A](
      expr: Ein[A],
      upstream: Option[Ein[A]],
      accum: scala.collection.mutable.Map[String, Ein[A]]
  ): Unit =
    expr match
      case Ein.Param(id, _, _) =>
        val grad = upstream.getOrElse(Ein.Ones(expr.outputDims))
        accum.get(id) match
          case Some(existing) => accum(id) = Ein.ElemAdd(existing, grad)
          case None           => accum(id) = grad

      case Ein.Input(_, _)  => ()
      case Ein.Fill(_, _)   => ()
      case Ein.Ones(_)      => ()
      case Ein.Zeros(_)     => ()

      case Ein.Contract(left, right) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val leftGrad = Ein.Contract(up, right)
        val rightGrad = Ein.Contract(up, left)
        backwardAccum(left, Some(leftGrad), accum)
        backwardAccum(right, Some(rightGrad), accum)

      case Ein.ElemAdd(left, right) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        backwardAccum(left, Some(up), accum)
        backwardAccum(right, Some(up), accum)

      case Ein.ElemSub(left, right) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        backwardAccum(left, Some(up), accum)
        val negUp = Ein.ElemSub(Ein.Zeros(up.outputDims), up)
        backwardAccum(right, Some(negUp), accum)

      case Ein.ElemMul(left, right) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val leftGrad = Ein.ElemMul(up, right)
        val rightGrad = Ein.ElemMul(up, left)
        backwardAccum(left, Some(leftGrad), accum)
        backwardAccum(right, Some(rightGrad), accum)

      case Ein.Activate(f, arg) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val argGrad = Ein.ElemMul(up, Ein.ActivateDeriv(f, arg))
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.ActivateDeriv(_, _) => ()

      case Ein.ReduceSum(arg, over) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val argGrad = Ein.Broadcast(up, arg.outputDims)
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.Broadcast(arg, targetDims) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val argNames = arg.outputDims.map(_.name).toSet
        val broadcastedNames = targetDims.map(_.name).filterNot(argNames.contains)
        val argGrad = Ein.ReduceSum(up, broadcastedNames)
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.Transpose(arg, perm) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val argNames = arg.outputDims.map(_.name)
        val argGrad = Ein.Transpose(up, argNames)
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.Softmax(arg, overDim) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val s = Ein.Softmax(arg, overDim)
        val dot = Ein.ElemMul(up, s)
        val sumDot = Ein.Broadcast(Ein.ReduceSum(dot, List(overDim)), arg.outputDims)
        val diff = Ein.ElemSub(up, sumDot)
        val argGrad = Ein.ElemMul(s, diff)
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.LogSoftmax(arg, overDim) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val s = Ein.Softmax(arg, overDim)
        val sumUp = Ein.Broadcast(Ein.ReduceSum(up, List(overDim)), arg.outputDims)
        val scaled = Ein.ElemMul(s, sumUp)
        val argGrad = Ein.ElemSub(up, scaled)
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.LayerNorm(arg, scale, bias, overDims, eps) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val normalized = Ein.LayerNorm(arg, scale, bias, overDims, eps)

        val biasGrad = Ein.ReduceSum(up, overDims)
        backwardAccum(bias, Some(biasGrad), accum)

        val scaleGrad = Ein.ReduceSum(Ein.ElemMul(up, normalized), overDims)
        backwardAccum(scale, Some(scaleGrad), accum)

        val argGrad = Ein.ElemMul(up, scale)
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.Reshape(arg, targetDims) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val argGrad = Ein.Reshape(up, arg.outputDims)
        backwardAccum(arg, Some(argGrad), accum)

      case Ein.Slice(arg, dim, from, to) =>
        // Gradient of slice is not representable in Ein algebra (needs scatter-pad)
        // Skip for now — use EinGrad.backward for numerical gradients of Slice expressions
        ()

      case Ein.Gather(table, indices, lookupDim) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val tableGrad = Ein.Scatter(up, indices, lookupDim, table.outputDims)
        backwardAccum(table, Some(tableGrad), accum)

      case Ein.Scatter(src, indices, lookupDim, tableDims) =>
        val up = upstream.getOrElse(Ein.Ones(expr.outputDims))
        val srcGrad = Ein.Gather(up, indices, lookupDim)
        backwardAccum(src, Some(srcGrad), accum)
