package palladium

object Loss:
  /**
   * Mean Squared Error loss: (1/n) * Σ(predicted - target)²
   *
   * @param predicted sequence of predicted values
   * @param target sequence of target values
   * @return MSE loss as a Value expression
   */
  def mse[A: NumberLike](
      predicted: Seq[Value[A]],
      target: Seq[Value[A]]
  ): Value[A] =
    require(
      predicted.length == target.length,
      s"predicted and target must have same length (got ${predicted.length} vs ${target.length})"
    )
    require(predicted.nonEmpty, "predicted and target cannot be empty")

    val num = summon[NumberLike[A]]
    val n = predicted.length

    // Compute Σ(predicted - target)²
    val squaredErrors = predicted.zip(target).map { case (pred, targ) =>
      val diff = pred - targ
      diff * diff
    }

    // Sum all squared errors
    val sum = squaredErrors.reduce(_ + _)

    // Divide by n to get mean
    sum / Value.Const(n)

  /**
   * Binary cross-entropy loss for binary classification
   * BCE = -(1/n) * Σ[target * log(predicted) + (1 - target) * log(1 - predicted)]
   *
   * @param predicted sequence of predicted probabilities (should be in [0,1])
   * @param target sequence of target labels (should be 0 or 1)
   * @return binary cross-entropy loss as a Value expression
   */
  def binaryCrossEntropy[A: NumberLike](
      predicted: Seq[Value[A]],
      target: Seq[Value[A]]
  ): Value[A] =
    require(
      predicted.length == target.length,
      s"predicted and target must have same length (got ${predicted.length} vs ${target.length})"
    )
    require(predicted.nonEmpty, "predicted and target cannot be empty")

    val num = summon[NumberLike[A]]
    val n = predicted.length
    val one = Value.Lit(num.fromInt(1))

    // Compute -[target * log(predicted) + (1 - target) * log(1 - predicted)]
    val losses = predicted.zip(target).map { case (pred, targ) =>
      val term1 = targ * pred.log
      val term2 = (one - targ) * (one - pred).log
      -(term1 + term2)
    }

    // Sum all losses
    val sum = losses.reduce(_ + _)

    // Divide by n to get mean
    sum / Value.Const(n)

  /**
   * Categorical cross-entropy loss for multi-class classification
   * CCE = -(1/n) * Σ_i Σ_j target[i,j] * log(predicted[i,j])
   *
   * @param predicted sequence of sequences (each inner seq is a probability distribution over classes)
   * @param target sequence of sequences (each inner seq is a one-hot encoded target)
   * @return categorical cross-entropy loss as a Value expression
   */
  def categoricalCrossEntropy[A: NumberLike](
      predicted: Seq[Seq[Value[A]]],
      target: Seq[Seq[Value[A]]]
  ): Value[A] =
    require(
      predicted.length == target.length,
      s"predicted and target must have same length (got ${predicted.length} vs ${target.length})"
    )
    require(predicted.nonEmpty, "predicted and target cannot be empty")
    require(
      predicted.forall(_.length == predicted.head.length),
      "all predicted samples must have same number of classes"
    )
    require(
      target.forall(_.length == target.head.length),
      "all target samples must have same number of classes"
    )
    require(
      predicted.head.length == target.head.length,
      s"predicted and target must have same number of classes (got ${predicted.head.length} vs ${target.head.length})"
    )

    val num = summon[NumberLike[A]]
    val n = predicted.length

    // Compute -Σ_i Σ_j target[i,j] * log(predicted[i,j])
    val losses = predicted.zip(target).map { case (predSample, targSample) =>
      predSample.zip(targSample).map { case (pred, targ) =>
        targ * pred.log
      }.reduce(_ + _)
    }

    // Sum all sample losses and negate
    val sum = -losses.reduce(_ + _)

    // Divide by n to get mean
    sum / Value.Const(n)

  object syntax:
    extension [A: NumberLike](predicted: Seq[Value[A]])
      def mse(target: Seq[Value[A]]): Value[A] = Loss.mse(predicted, target)
      def binaryCrossEntropy(target: Seq[Value[A]]): Value[A] =
        Loss.binaryCrossEntropy(predicted, target)

    extension [A: NumberLike](predicted: Seq[Seq[Value[A]]])
      def categoricalCrossEntropy(target: Seq[Seq[Value[A]]]): Value[A] =
        Loss.categoricalCrossEntropy(predicted, target)
