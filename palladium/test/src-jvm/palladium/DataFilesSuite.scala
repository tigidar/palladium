package palladium

import scala.collection.SortedMap
import palladium.ein.{TensorData, Dim}
import shapesdsl.dsl.*
import shapesdsl.HeatmapImage

case class Bigram(
    counts: TensorData[Int],
    rowToChar: Map[Int, Char],
    colToChar: Map[Int, Char],
    charToRow: Map[Char, Int],
    charToCol: Map[Char, Int]
):
  def apply(c1: Char, c2: Char): Int =
    counts(Array(charToRow(c1), charToCol(c2)))
  def increment(c1: Char, c2: Char): Unit =
    val i = charToRow(c1)
    val j = charToCol(c2)
    counts.update(Array(i, j), counts(Array(i, j)) + 1)

object Bigram:

  /** Build a bigram from words. Rows = {^, a..z}, Columns = {a..z, $}.
    * ^ only starts words (row), $ only ends words (column).
    */
  def fromWords(words: Seq[String]): Bigram =
    val chars = words.foldLeft(Set.empty[Char])((acc, w) => acc ++ w.toSet)
    val sorted = chars.toList.sorted

    // rows: ^ first, then a..z (^ can start a word, letters can follow letters)
    val rowChars = '^' :: sorted
    val charToRow = rowChars.zipWithIndex.toMap
    val rowToChar = charToRow.map((c, i) => i -> c)

    // cols: a..z, then $ (letters can follow, $ ends a word)
    val colChars = sorted :+ '$'
    val charToCol = colChars.zipWithIndex.toMap
    val colToChar = charToCol.map((c, i) => i -> c)

    val nRows = rowChars.size
    val nCols = colChars.size
    val from = Dim("from", nRows)
    val to = Dim("to", nCols)
    val N = TensorData.zeros[Int](List(from, to))

    val bigram = Bigram(N, rowToChar, colToChar, charToRow, charToCol)

    words.foreach { word =>
      val padded = '^' + word.trim() + '$'
      padded.zip(padded.tail).foreach { (c1, c2) =>
        bigram.increment(c1, c2)
      }
    }

    bigram

class DataFilesSuite extends munit.FunSuite:

  // Mill sandboxes test working dirs; use MILL_WORKSPACE_ROOT env var to find project root
  val projectRoot: os.Path =
    sys.env
      .get("MILL_WORKSPACE_ROOT")
      .map(os.Path(_))
      .getOrElse(os.pwd)

  val path = projectRoot / "data" / "names.txt"
  val lines = os.read.lines(path)

  test("we are able to load names.txt") {
    assert(
      lines.size == 32033,
      s"Expected more than 10 lines, got ${lines.size}"
    )
    val wordSizes = lines.map(_.size)
    assert(wordSizes.max == 15, "Expect longest word to be 15")
    assert(wordSizes.min == 2, "Expect shortest word to be 2")
  }

  test("basic bigram") {
    val bigram = Bigram.fromWords(lines)

    val data = bigram.counts.toRows

    val hm = heatmap(data).withCellLabels(
      data.zipWithIndex.map { (row, rowIndex) =>
        row.zipWithIndex.map { (col, colIndex) =>
          val first = bigram.rowToChar(rowIndex)
          val second = bigram.colToChar(colIndex)
          s"$first$second\n$col"
        }
      }
    )

    val png = HeatmapImage.toPng(hm)
    os.write.over(projectRoot / "heatmap.png", png)

    // ^ row should have counts (names start with letters)
    assert(bigram('^', 'a') > 0, "some names should start with 'a'")
  }
