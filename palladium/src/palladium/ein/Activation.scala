package palladium.ein

import palladium.NumberLike

enum Activation:
  case ReLU, Sigmoid, Tanh, GELU, Swish

object Activation:
  def apply[A: NumberLike](f: Activation, x: A): A =
    val num = summon[NumberLike[A]]
    f match
      case Activation.ReLU    => num.relu(x)
      case Activation.Sigmoid =>
        val one = num.fromInt(1)
        num.div(one, num.plus(one, num.exp(num.negate(x))))
      case Activation.Tanh =>
        val ex = num.exp(x)
        val enx = num.exp(num.negate(x))
        num.div(num.minus(ex, enx), num.plus(ex, enx))
      case Activation.GELU =>
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        val half = num.div(num.fromInt(1), num.fromInt(2))
        val coeff = num.div(num.fromInt(44715), num.fromInt(1000000)) // 0.044715
        val pi = num.div(num.fromInt(355), num.fromInt(113)) // pi ≈ 355/113
        val sqrtTwoOverPi = num.sqrt(num.div(num.fromInt(2), pi))
        val x3 = num.times(x, num.times(x, x))
        val inner = num.times(sqrtTwoOverPi, num.plus(x, num.times(coeff, x3)))
        val tanhInner = Activation(Activation.Tanh, inner)
        num.times(half, num.times(x, num.plus(num.fromInt(1), tanhInner)))
      case Activation.Swish =>
        // Swish(x) = x * sigmoid(x)
        num.times(x, Activation(Activation.Sigmoid, x))

  def deriv[A: NumberLike](f: Activation, x: A): A =
    val num = summon[NumberLike[A]]
    val one = num.fromInt(1)
    f match
      case Activation.ReLU => num.step(x)
      case Activation.Sigmoid =>
        val s = Activation(Activation.Sigmoid, x)
        num.times(s, num.minus(one, s))
      case Activation.Tanh =>
        val t = Activation(Activation.Tanh, x)
        num.minus(one, num.times(t, t))
      case Activation.GELU =>
        // d/dx GELU via analytic formula:
        // GELU'(x) = 0.5*(1+tanh(u)) + 0.5*x*sech²(u)*du/dx
        // where u = sqrt(2/pi)*(x + 0.044715*x³), du/dx = sqrt(2/pi)*(1 + 3*0.044715*x²)
        val half = num.div(num.fromInt(1), num.fromInt(2))
        val coeff = num.div(num.fromInt(44715), num.fromInt(1000000)) // 0.044715
        val coeff3 = num.div(num.fromInt(134145), num.fromInt(1000000)) // 3 * 0.044715
        val pi = num.div(num.fromInt(355), num.fromInt(113))
        val sqrtTwoOverPi = num.sqrt(num.div(num.fromInt(2), pi))
        val x3 = num.times(x, num.times(x, x))
        val u = num.times(sqrtTwoOverPi, num.plus(x, num.times(coeff, x3)))
        val tanhU = Activation(Activation.Tanh, u)
        val duDx = num.times(sqrtTwoOverPi, num.plus(one, num.times(coeff3, num.times(x, x))))
        val sech2U = num.minus(one, num.times(tanhU, tanhU))
        val term1 = num.times(half, num.plus(one, tanhU))
        val term2 = num.times(half, num.times(x, num.times(sech2U, duDx)))
        num.plus(term1, term2)
      case Activation.Swish =>
        // d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //                        = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        val s = Activation(Activation.Sigmoid, x)
        num.times(s, num.plus(one, num.times(x, num.minus(one, s))))
