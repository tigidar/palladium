package palladium.blas

import palladium.codegen.*
import palladium.ein.*

/** High-level entry point: Ein expression → BLAS-accelerated execution.
  *
  * Pipeline: Ein[Double] → Lower.lower → LowProgram → BlasExec.run
  *
  * This mirrors the CGen pipeline but executes natively instead of
  * generating C source code. Uses CBLAS for matrix operations and
  * arena-allocated Ptr[Double] buffers for zero-GC-pressure computation.
  */
object BlasGen:

  /** Evaluate an Ein expression using BLAS, returning a TensorData result.
    *
    * @param expr the tensor expression to evaluate
    * @param feed input name → TensorData mappings
    * @return the computed output as TensorData[Double]
    */
  def eval(expr: Ein[Double], feed: Map[String, TensorData[Double]]): TensorData[Double] =
    val prog = Lower.lower(expr)

    val paramData: Map[String, Array[Double]] = collectParams(expr)
    val inputData: Map[String, Array[Double]] = feed.map { (name, td) =>
      name -> td.data.map(_.toDouble)
    }

    val results = BlasExec.run(prog, paramData, inputData)
    val outRef = prog.outputs.head
    TensorData.fromArray(
      outRef.shape.zipWithIndex.map((s, i) => Dim(s"d$i", s)),
      results.head
    )

  /** Evaluate and return raw output arrays (for multiple outputs). */
  def evalRaw(
      expr: Ein[Double],
      feed: Map[String, TensorData[Double]]
  ): Vector[Array[Double]] =
    val prog = Lower.lower(expr)
    val paramData = collectParams(expr)
    val inputData = feed.map((name, td) => name -> td.data.map(_.toDouble))
    BlasExec.run(prog, paramData, inputData)

  /** Evaluate a pre-lowered program directly (for cases where you
    * want to control the lowering step yourself).
    */
  def evalProgram(
      prog: LowProgram,
      paramData: Map[String, Array[Double]],
      inputData: Map[String, Array[Double]]
  ): Vector[Array[Double]] =
    BlasExec.run(prog, paramData, inputData)

  /** Recursively collect param data from Ein expression tree. */
  private def collectParams(expr: Ein[Double]): Map[String, Array[Double]] =
    expr match
      case Ein.Param(id, _, data)      => Map(id -> data.data.map(_.toDouble))
      case Ein.Contract(l, r)          => collectParams(l) ++ collectParams(r)
      case Ein.ElemAdd(l, r)           => collectParams(l) ++ collectParams(r)
      case Ein.ElemSub(l, r)           => collectParams(l) ++ collectParams(r)
      case Ein.ElemMul(l, r)           => collectParams(l) ++ collectParams(r)
      case Ein.Activate(_, arg)        => collectParams(arg)
      case Ein.ActivateDeriv(_, arg)   => collectParams(arg)
      case Ein.ReduceSum(arg, _)       => collectParams(arg)
      case Ein.Broadcast(arg, _)       => collectParams(arg)
      case Ein.Transpose(arg, _)       => collectParams(arg)
      case Ein.Softmax(arg, _)         => collectParams(arg)
      case Ein.LogSoftmax(arg, _)      => collectParams(arg)
      case Ein.LayerNorm(arg, s, b, _, _) =>
        collectParams(arg) ++ collectParams(s) ++ collectParams(b)
      case Ein.Reshape(arg, _)         => collectParams(arg)
      case Ein.Slice(arg, _, _, _)     => collectParams(arg)
      case Ein.Gather(t, _, _)         => collectParams(t)
      case Ein.Scatter(s, _, _, _)     => collectParams(s)
      case _                           => Map.empty
