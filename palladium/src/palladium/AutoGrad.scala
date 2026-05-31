package palladium

case class Traced[A](value: A, grads: Map[String, A])

object AutoGrad:
  /** Evaluate an expression and compute gradients for all variables */
  def trace[A: NumberLike](expr: Value[A]): Traced[A] =
    Traced(expr.eval, Grad.backward(expr))

  object syntax:
    extension [A: NumberLike](expr: Value[A])
      def trace: Traced[A] = AutoGrad.trace(expr)
