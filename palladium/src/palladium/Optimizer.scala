package palladium

/** Optimizers for updating model parameters based on computed gradients */
trait Optimizer[A]:
  /** Update parameters given their gradients
    *
    * @param params current parameter values
    * @param grads gradient for each parameter
    * @return updated parameter values
    */
  def step(params: Map[String, A], grads: Map[String, A]): Map[String, A]

object Optimizer:
  /** Stochastic Gradient Descent optimizer
    *
    * Updates parameters using: param_new = param_old - learning_rate * gradient
    *
    * @param learningRate step size for parameter updates
    */
  case class SGD[A: NumberLike](learningRate: A) extends Optimizer[A]:
    def step(params: Map[String, A], grads: Map[String, A]): Map[String, A] =
      val num = summon[NumberLike[A]]
      params.map { case (name, param) =>
        val grad = grads.getOrElse(
          name,
          num.fromInt(0)
        )
        val update = num.times(learningRate, grad)
        name -> num.minus(param, update)
      }

  /** AdamW optimizer with decoupled weight decay
    *
    * Maintains per-parameter first and second moment estimates.
    * Updates: m = β1*m + (1-β1)*g, v = β2*v + (1-β2)*g², then
    * param_new = param - lr * (m̂/(√v̂ + ε) + weightDecay * param)
    */
  class AdamW[A: NumberLike](
      val learningRate: A,
      val beta1: A,
      val beta2: A,
      val epsilon: A,
      val weightDecay: A
  ) extends Optimizer[A]:
    private val num = summon[NumberLike[A]]
    private var t: Int = 0
    private val m = scala.collection.mutable.Map[String, A]()
    private val v = scala.collection.mutable.Map[String, A]()

    def step(params: Map[String, A], grads: Map[String, A]): Map[String, A] =
      val zero = num.fromInt(0)
      val one = num.fromInt(1)
      t += 1

      params.map { case (name, param) =>
        val grad = grads.getOrElse(name, zero)

        // Update first moment: m_t = β1 * m_{t-1} + (1 - β1) * g
        val mPrev = m.getOrElse(name, zero)
        val mT = num.plus(num.times(beta1, mPrev), num.times(num.minus(one, beta1), grad))
        m(name) = mT

        // Update second moment: v_t = β2 * v_{t-1} + (1 - β2) * g²
        val vPrev = v.getOrElse(name, zero)
        val vT = num.plus(num.times(beta2, vPrev), num.times(num.minus(one, beta2), num.times(grad, grad)))
        v(name) = vT

        // Bias correction
        val mHat = num.div(mT, num.minus(one, num.pow(beta1, num.fromInt(t))))
        val vHat = num.div(vT, num.minus(one, num.pow(beta2, num.fromInt(t))))

        // Adam update + decoupled weight decay
        val adamUpdate = num.div(mHat, num.plus(num.sqrt(vHat), epsilon))
        val wdUpdate = num.times(weightDecay, param)
        name -> num.minus(param, num.times(learningRate, num.plus(adamUpdate, wdUpdate)))
      }

  object AdamW:
    def default[A: NumberLike](learningRate: A): AdamW[A] =
      val num = summon[NumberLike[A]]
      val one = num.fromInt(1)
      new AdamW[A](
        learningRate = learningRate,
        beta1 = num.minus(one, num.div(one, num.fromInt(10))),           // 0.9
        beta2 = num.minus(one, num.div(one, num.fromInt(1000))),         // 0.999
        epsilon = num.div(one, num.fromInt(100000000)),                   // 1e-8
        weightDecay = num.div(one, num.fromInt(100))                      // 0.01
      )

  object syntax:
    extension [A](opt: Optimizer[A])
      /** Apply optimizer step to update parameters */
      def apply(params: Map[String, A], grads: Map[String, A]): Map[String, A] =
        opt.step(params, grads)
