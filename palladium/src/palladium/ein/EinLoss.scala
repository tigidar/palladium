package palladium.ein

import palladium.NumberLike
import scala.reflect.ClassTag

object EinLoss:

  /** Cross-entropy loss: -mean(sum(targets * log_softmax(logits), classDim))
    *
    * Computes: reduceSum(-targets * logSoftmax(logits)) over all dimensions.
    * Targets should be one-hot encoded with the same shape as logits.
    *
    * @param logits raw model outputs (before softmax)
    * @param targets one-hot encoded targets (same shape as logits)
    * @param classDim dimension name to apply softmax over (the class dimension)
    * @return scalar loss (all dims reduced)
    */
  def crossEntropy[A: NumberLike: ClassTag](logits: Ein[A], targets: Ein[A], classDim: String): Ein[A] =
    val num = summon[NumberLike[A]]
    val logProbs = Ein.LogSoftmax(logits, classDim)
    val pointwise = Ein.ElemMul(targets, logProbs)
    // Sum over all dims to get scalar
    val allDims = logits.outputDims.map(_.name)
    val total = Ein.ReduceSum(pointwise, allDims)
    // Negate: CE = -sum(targets * log_probs)
    // Subtract from zero: 0 - total
    val zero = Ein.Fill(num.fromInt(0), Nil)
    Ein.ElemSub(zero, total)

  /** MSE loss at tensor level: sum((predicted - target)^2)
    *
    * @param predicted model predictions
    * @param target ground truth (same shape as predicted)
    * @return scalar loss (all dims reduced)
    */
  def mse[A](predicted: Ein[A], target: Ein[A]): Ein[A] =
    val diff = Ein.ElemSub(predicted, target)
    val sq = Ein.ElemMul(diff, diff)
    val allDims = predicted.outputDims.map(_.name)
    Ein.ReduceSum(sq, allDims)
