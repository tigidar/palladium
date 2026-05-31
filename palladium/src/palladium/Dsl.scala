package palladium

import sourcecode.Name
import scala.language.implicitConversions

object Dsl:
  // Symbolic lift: val x = 10.0.^ → Value.Var("x", 10.0)
  // The ^ symbolizes "lifting" the value into the DSL
  extension [A: NumberLike](value: A)
    def ^(using name: Name): Value[A] = Value.Var(name.value, value)
    def toVar(using name: Name): Value[A] = Value.Var(name.value, value)

  // Free-standing functions: tanh(expr), exp(expr), log(expr)
  def tanh[A](value: Value[A]): Value[A] = value.tanh
  def exp[A](value: Value[A]): Value[A] = value.exp
  def log[A](value: Value[A]): Value[A] = value.log

  // Implicit conversion: allows x + 3.0 instead of x + Value(3.0)
  given numberToValue[A: NumberLike]: Conversion[A, Value[A]] = Value.Lit(_)
