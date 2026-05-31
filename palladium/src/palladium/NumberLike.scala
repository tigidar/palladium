package palladium

trait NumberLike[A]:
  def plus(x: A, y: A): A
  def minus(x: A, y: A): A
  def times(x: A, y: A): A
  def div(x: A, y: A): A
  def pow(x: A, exp: A): A
  def negate(x: A): A
  def log(x: A): A
  def exp(x: A): A
  def sin(x: A): A
  def cos(x: A): A
  def abs(x: A): A
  def sqrt(x: A): A
  def signum(x: A): A   // sign(x): -1, 0, or 1
  def relu(x: A): A     // max(0, x)
  def step(x: A): A     // Heaviside: 1 if x > 0, 0 otherwise
  def max(x: A, y: A): A
  def negativeInfinity: A
  def fromInt(n: Int): A

object NumberLike:
  given NumberLike[Double] with
    def plus(x: Double, y: Double): Double = x + y
    def minus(x: Double, y: Double): Double = x - y
    def times(x: Double, y: Double): Double = x * y
    def div(x: Double, y: Double): Double = x / y
    def pow(x: Double, exp: Double): Double = math.pow(x, exp)
    def negate(x: Double): Double = -x
    def log(x: Double): Double = math.log(x)
    def exp(x: Double): Double = math.exp(x)
    def sin(x: Double): Double = math.sin(x)
    def cos(x: Double): Double = math.cos(x)
    def abs(x: Double): Double = math.abs(x)
    def sqrt(x: Double): Double = math.sqrt(x)
    def signum(x: Double): Double = math.signum(x)
    def relu(x: Double): Double = math.max(0.0, x)
    def step(x: Double): Double = if x > 0.0 then 1.0 else 0.0
    def max(x: Double, y: Double): Double = math.max(x, y)
    def negativeInfinity: Double = Double.NegativeInfinity
    def fromInt(n: Int): Double = n.toDouble

  given NumberLike[Int] with
    def plus(x: Int, y: Int): Int = x + y
    def minus(x: Int, y: Int): Int = x - y
    def times(x: Int, y: Int): Int = x * y
    def div(x: Int, y: Int): Int = x / y
    def pow(x: Int, exp: Int): Int = math.pow(x.toDouble, exp.toDouble).toInt
    def negate(x: Int): Int = -x
    def log(x: Int): Int = math.log(x.toDouble).toInt
    def exp(x: Int): Int = math.exp(x.toDouble).toInt
    def sin(x: Int): Int = math.sin(x.toDouble).toInt
    def cos(x: Int): Int = math.cos(x.toDouble).toInt
    def abs(x: Int): Int = math.abs(x)
    def sqrt(x: Int): Int = math.sqrt(x.toDouble).toInt
    def signum(x: Int): Int = x.sign
    def relu(x: Int): Int = math.max(0, x)
    def step(x: Int): Int = if x > 0 then 1 else 0
    def max(x: Int, y: Int): Int = math.max(x, y)
    def negativeInfinity: Int = Int.MinValue
    def fromInt(n: Int): Int = n

  given NumberLike[Float] with
    def plus(x: Float, y: Float): Float = x + y
    def minus(x: Float, y: Float): Float = x - y
    def times(x: Float, y: Float): Float = x * y
    def div(x: Float, y: Float): Float = x / y
    def pow(x: Float, exp: Float): Float = math.pow(x.toDouble, exp.toDouble).toFloat
    def negate(x: Float): Float = -x
    def log(x: Float): Float = math.log(x.toDouble).toFloat
    def exp(x: Float): Float = math.exp(x.toDouble).toFloat
    def sin(x: Float): Float = math.sin(x.toDouble).toFloat
    def cos(x: Float): Float = math.cos(x.toDouble).toFloat
    def abs(x: Float): Float = math.abs(x)
    def sqrt(x: Float): Float = math.sqrt(x.toDouble).toFloat
    def signum(x: Float): Float = math.signum(x)
    def relu(x: Float): Float = math.max(0.0f, x)
    def step(x: Float): Float = if x > 0.0f then 1.0f else 0.0f
    def max(x: Float, y: Float): Float = math.max(x, y)
    def negativeInfinity: Float = Float.NegativeInfinity
    def fromInt(n: Int): Float = n.toFloat
