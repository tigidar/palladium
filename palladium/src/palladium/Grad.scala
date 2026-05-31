package palladium

object Grad:
  /** Compute gradients for all variables in the expression via reverse-mode autodiff */
  def backward[A: NumberLike](expr: Value[A]): Map[String, A] =
    val num = summon[NumberLike[A]]
    backwardAccum(expr, num.fromInt(1), Map.empty)

  private def backwardAccum[A](
      expr: Value[A],
      upstream: A,
      accum: Map[String, A]
  )(using num: NumberLike[A]): Map[String, A] =
    import Value.*
    expr match
      case Var(id, _) =>
        val existing = accum.getOrElse(id, num.fromInt(0))
        accum.updated(id, num.plus(existing, upstream))

      case Lit(_) | Const(_) =>
        accum // constants have no gradient

      case Add(l, r) =>
        // d/dl (l + r) = 1, d/dr (l + r) = 1
        val a1 = backwardAccum(l, upstream, accum)
        backwardAccum(r, upstream, a1)

      case Sub(l, r) =>
        // d/dl (l - r) = 1, d/dr (l - r) = -1
        val a1 = backwardAccum(l, upstream, accum)
        backwardAccum(r, num.negate(upstream), a1)

      case Mul(l, r) =>
        // d/dl (l * r) = r, d/dr (l * r) = l
        val a1 = backwardAccum(l, num.times(upstream, r.eval), accum)
        backwardAccum(r, num.times(upstream, l.eval), a1)

      case Div(l, r) =>
        // d/dl (l / r) = 1/r
        // d/dr (l / r) = -l / r^2
        val rVal = r.eval
        val lVal = l.eval
        val a1 = backwardAccum(l, num.div(upstream, rVal), accum)
        val rGrad = num.negate(
          num.div(num.times(upstream, lVal), num.times(rVal, rVal))
        )
        backwardAccum(r, rGrad, a1)

      case FastDiv(l, r) =>
        // FastDiv(a, b) = a * b^(-1)
        // d/da = b^(-1)
        // d/db = -a * b^(-2)
        val rVal = r.eval
        val lVal = l.eval
        val rInv = num.pow(rVal, num.negate(num.fromInt(1)))
        val a1 = backwardAccum(l, num.times(upstream, rInv), accum)
        val rGrad = num.negate(
          num.times(upstream, num.times(lVal, num.pow(rVal, num.negate(num.fromInt(2)))))
        )
        backwardAccum(r, rGrad, a1)

      case Pow(base, exp) =>
        // d/dbase (base^exp) = exp * base^(exp-1)
        // d/dexp (base^exp) = base^exp * log(base)
        val baseVal = base.eval
        val expVal = exp.eval
        val powVal = num.pow(baseVal, expVal)

        val baseGrad = num.times(
          upstream,
          num.times(expVal, num.pow(baseVal, num.minus(expVal, num.fromInt(1))))
        )
        val a1 = backwardAccum(base, baseGrad, accum)

        val expGrad = num.times(upstream, num.times(powVal, num.log(baseVal)))
        backwardAccum(exp, expGrad, a1)

      case Neg(v) =>
        // d/dv (-v) = -1
        backwardAccum(v, num.negate(upstream), accum)

      case Log(v) =>
        // d/dv log(v) = 1/v
        val vGrad = num.div(upstream, v.eval)
        backwardAccum(v, vGrad, accum)

      case Exp(v) =>
        // d/dv exp(v) = exp(v)
        val vGrad = num.times(upstream, num.exp(v.eval))
        backwardAccum(v, vGrad, accum)

      case Tanh(v) =>
        // d/dv tanh(v) = 1 - tanh(v)^2
        val vVal = v.eval
        val ev = num.exp(vVal)
        val emv = num.exp(num.negate(vVal))
        val tanhVal = num.div(num.minus(ev, emv), num.plus(ev, emv))
        val vGrad = num.times(upstream, num.minus(num.fromInt(1), num.times(tanhVal, tanhVal)))
        backwardAccum(v, vGrad, accum)

      case Sigmoid(v) =>
        // d/dv σ(v) = σ(v) * (1 - σ(v))
        val vVal = v.eval
        val s = num.div(num.fromInt(1), num.plus(num.fromInt(1), num.exp(num.negate(vVal))))
        val vGrad = num.times(upstream, num.times(s, num.minus(num.fromInt(1), s)))
        backwardAccum(v, vGrad, accum)

      case Relu(v) =>
        // d/dv relu(v) = step(v) (1 if v > 0, 0 otherwise)
        val vGrad = num.times(upstream, num.step(v.eval))
        backwardAccum(v, vGrad, accum)

      case Sin(v) =>
        // d/dv sin(v) = cos(v)
        val vGrad = num.times(upstream, num.cos(v.eval))
        backwardAccum(v, vGrad, accum)

      case Cos(v) =>
        // d/dv cos(v) = -sin(v)
        val vGrad = num.times(upstream, num.negate(num.sin(v.eval)))
        backwardAccum(v, vGrad, accum)

      case Abs(v) =>
        // d/dv |v| = sign(v)
        val vGrad = num.times(upstream, num.signum(v.eval))
        backwardAccum(v, vGrad, accum)

      case Step(_) | Signum(_) =>
        accum

  object syntax:
    extension [A: NumberLike](expr: Value[A])
      def backward: Map[String, A] = Grad.backward(expr)
