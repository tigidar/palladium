package palladium

enum Value[A]:
  case Var(id: String, data: A)
  case Lit(data: A)
  case Const(n: Int)
  case Add(left: Value[A], right: Value[A])
  case Sub(left: Value[A], right: Value[A])
  case Mul(left: Value[A], right: Value[A])
  case Div(left: Value[A], right: Value[A])
  case FastDiv(left: Value[A], right: Value[A])
  case Pow(base: Value[A], exponent: Value[A])
  case Neg(value: Value[A])
  case Log(value: Value[A])
  case Exp(value: Value[A])
  case Tanh(value: Value[A])
  case Sigmoid(value: Value[A])  // σ(x) = 1 / (1 + exp(-x))
  case Relu(value: Value[A])     // max(0, x)
  case Sin(value: Value[A])      // sin(x)
  case Cos(value: Value[A])      // cos(x)
  case Abs(value: Value[A])      // |x|
  case Step(value: Value[A])     // Heaviside: 1 if x > 0, 0 otherwise
  case Signum(value: Value[A])   // sign(x): -1, 0, or 1

  def +(other: Value[A]): Value[A] = Add(this, other)
  def -(other: Value[A]): Value[A] = Sub(this, other)
  def *(other: Value[A]): Value[A] = Mul(this, other)
  def /(other: Value[A]): Value[A] = Div(this, other)
  def /~(other: Value[A]): Value[A] = FastDiv(this, other)
  def ~^(exponent: Value[A]): Value[A] = Pow(this, exponent)
  def pow(exponent: Value[A]): Value[A] = Pow(this, exponent)
  def unary_- : Value[A] = Neg(this)
  def negate: Value[A] = Neg(this)
  def log: Value[A] = Log(this)
  def exp: Value[A] = Exp(this)
  def tanh: Value[A] = Tanh(this)
  def sigmoid: Value[A] = Sigmoid(this)
  def relu: Value[A] = Relu(this)
  def sin: Value[A] = Sin(this)
  def cos: Value[A] = Cos(this)
  def abs: Value[A] = Abs(this)
  def step: Value[A] = Step(this)
  def signum: Value[A] = Signum(this)

  def eval(using num: NumberLike[A]): A =
    this match
      case Var(_, data)     => data
      case Lit(data)        => data
      case Const(n)         => num.fromInt(n)
      case Add(left, right) => num.plus(left.eval, right.eval)
      case Sub(left, right) => num.minus(left.eval, right.eval)
      case Mul(left, right) => num.times(left.eval, right.eval)
      case Div(left, right) => num.div(left.eval, right.eval)
      case FastDiv(left, right) =>
        num.times(left.eval, num.pow(right.eval, num.negate(num.fromInt(1))))
      case Pow(base, exponent) => num.pow(base.eval, exponent.eval)
      case Neg(value)       => num.negate(value.eval)
      case Log(value)       => num.log(value.eval)
      case Exp(value)       => num.exp(value.eval)
      case Tanh(v) =>
        val vVal = v.eval
        val ev = num.exp(vVal)
        val emv = num.exp(num.negate(vVal))
        num.div(num.minus(ev, emv), num.plus(ev, emv))
      case Sigmoid(v) =>
        // σ(x) = 1 / (1 + exp(-x))
        val vVal = v.eval
        num.div(num.fromInt(1), num.plus(num.fromInt(1), num.exp(num.negate(vVal))))
      case Relu(v) =>
        // max(0, x)
        num.relu(v.eval)
      case Sin(v) =>
        num.sin(v.eval)
      case Cos(v) =>
        num.cos(v.eval)
      case Abs(v) =>
        // |x|
        num.abs(v.eval)
      case Step(v) =>
        num.step(v.eval)
      case Signum(v) =>
        num.signum(v.eval)

object Value:
  def apply[A](data: A): Value[A] = Lit(data)
  def const[A](n: Int): Value[A] = Const(n)
  def variable[A](id: String, data: A): Value[A] = Var(id, data)
