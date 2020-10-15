import org.apache.commons.csv.CSVFormat
import org.apache.commons.csv.CSVRecord
import java.io.FileReader
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

fun readFile(filename: String): List<List<Double>> {
    val fr = FileReader(filename)
    val records: Iterable<CSVRecord> = CSVFormat.DEFAULT.parse(fr)
    val data = mutableListOf<List<Double>>()
    for (record in records) {
        val row = mutableListOf<Double>()
        for (s in record) {
            row.add(s.toDouble())
        }

        data.add(row)
    }

    return data
}

fun List<List<Double>>.col(i: Int): List<Double> = map { row -> row[i] }

fun List<List<Double>>.cols(): List<List<Double>> = first().mapIndexed { index, _ -> col(index) }

fun List<List<Double>>.transposed() = cols()

fun List<Double>.mean(): Double = average()

fun List<Double>.std(): Double {
    var sum = 0.0
    val mean = this.mean()
    for (number in this) {
        sum += (number - mean).pow(2)
    }

    return sqrt(sum / size)
}

fun List<Double>.standardized(): List<Double> {
    val mean = mean()
    val std = std()
    return if (std == 0.0)
        map { 0.0 }
    else
        map { (it - mean) / std }
}

fun printSizeAndMeanAndStd(cols: List<List<Double>>) {
    println("rows: ${cols.first().size}")
    println("cols: ${cols.size}")
    println("F#\tMean\t\t\t\tSTD")
    for ((i, col) in cols.withIndex()) {
        println("F$i\t${col.mean()}\t${col.std()}")
    }
}

fun `partial derivative of MSE with respect to w_k`(
        xs: List<List<Double>>,
        y: List<Double>,
        w: List<Double>,
        k: Int
): Double {
    var sum = 0.0
    val n = xs.size
    for ((i, x) in xs.withIndex()) {
        var y2 = 0.0
        for ((j, d) in x.withIndex()) {
            y2 += w[j] * d
        }

        val error = y2 - y[i]
        sum += error * x[k]
    }

    return sum / n
}

fun `grad MSE`(xs: List<List<Double>>, y: List<Double>, w: List<Double>): List<Double> =
        w.mapIndexed { k, _ -> `partial derivative of MSE with respect to w_k`(xs, y, w, k) }

fun `stohastic gradient descent`(xs: List<List<Double>>, y: List<Double>, rate: Double, maxEpoch: Int, stopOnSmallR2Changes: Boolean = false, r2ChangeThreshold: Double = 1e-5): List<Double> {
    val tss = `total sum of squares`(y)
    val w = xs.first().mapIndexed { j, _ -> if (j == 0) 1.0 else 0.0 }.toMutableList()
    val intRange = 0 until w.size
    var oldR2: Double? = null
    for (epoch in 0 until maxEpoch) {
        val timeMs = measureTimeMillis {
            val k = intRange.random()
            require(k < w.size)
            val gradK = `partial derivative of MSE with respect to w_k`(xs, y, w, k)
            w[k] = w[k] - rate * gradK
        }

        if (epoch % 100 == 0) {
            val yPred = xs.map { x -> x.reduceIndexed { j, acc, d -> acc + d * w[j] } }

            val r2 = `coefficient of determination`(y, yPred, tss)

            if (oldR2 != null && stopOnSmallR2Changes && abs(oldR2 - r2) < r2ChangeThreshold) {
                println("r2 change is too small. Stopping on epoch $epoch. RMSE: ${rmse(xs, y, w)}. R2: $r2")
                return w
            }

            oldR2 = r2
        }
    }

    println("Exited after processed all epochs. RMSE: ${rmse(xs, y, w)}. R2: $oldR2")
    return w
}

fun `gradient descent`(xs: List<List<Double>>, y: List<Double>, rate: Double, maxEpoch: Int): List<Double> {
    val tss = `total sum of squares`(y)
    var w = xs.first().mapIndexed { j, _ -> if (j == 0) 1.0 else 0.0 }
    for (epoch in 0 until maxEpoch) {
        var mse = 0.0
        var rmse = 0.0
        var r2 = 0.0
        var gradTimeMs = 0L
        var mseTimeMs = 0L
        var rmseTimeMs = 0L
        var r2TimeMs = 0L
        val totalTimeMs = measureTimeMillis {
            var `grad MSE`: List<Double> = emptyList()
            gradTimeMs = measureTimeMillis { `grad MSE` = `grad MSE`(xs, y, w) }
            w = w.mapIndexed { j, d -> d - rate * `grad MSE`[j] }

            val yPred = xs.map { x -> x.reduceIndexed { j, acc, d -> acc + d * w[j] } }

            mseTimeMs = measureTimeMillis { mse = mse(xs, y, w) }
            rmseTimeMs = measureTimeMillis { rmse = rmse(xs, y, w) }
            r2TimeMs = measureTimeMillis { r2 = `coefficient of determination`(y, yPred, tss) }
        }

        println("epoch $epoch in $totalTimeMs ms. MSE: $mse. RMSE: $rmse. R2: $r2. GradTimeMs: $gradTimeMs. MSETimeMs: $mseTimeMs. RMSETimeMs: $rmseTimeMs. R2TimeMs: $r2TimeMs")
    }

    return w
}

fun rmse(xs: List<List<Double>>, y: List<Double>, w: List<Double>): Double {
    var sum = 0.0
    val n = xs.size
    for ((i, x) in xs.withIndex()) {
        var y2 = 0.0
        for ((j, d) in x.withIndex()) {
            y2 += w[j] * d
        }

        val errorSquared = (y2 - y[i]).pow(2)
        sum += errorSquared
    }

    return sqrt(sum / n)
}

fun mse(xs: List<List<Double>>, y: List<Double>, w: List<Double>): Double {
    var sum = 0.0
    val n = xs.size
    for ((i, x) in xs.withIndex()) {
        var y2 = 0.0
        for ((j, d) in x.withIndex()) {
            y2 += w[j] * d
        }

        val errorSquared = (y2 - y[i]).pow(2)
        sum += errorSquared
    }

    return sum / n
}

fun `total sum of squares`(y: List<Double>): Double {
    val mean = y.mean()
    var sum = 0.0
    for (d in y) {
        sum += (d - mean).pow(2)
    }

    return sum
}

fun `residual sum of squares`(y: List<Double>, yPred: List<Double>): Double {
    var sum = 0.0
    for ((i, d) in y.withIndex()) {
        sum += (d - yPred[i]).pow(2)
    }

    return sum
}

fun `coefficient of determination`(y: List<Double>, yPred: List<Double>, tss: Double): Double {
    val rss = `residual sum of squares`(y, yPred)

    return 1.0 - rss / tss
}

fun List<List<Double>>.shuffledAndChunked(k: Int = 5) = shuffled().windowed(size / k, size / k)

fun standardizeOnlyXsAndReturnCols(cols: List<List<Double>>): List<List<Double>> =
        cols.mapIndexed { j, col ->
            if (j != cols.size - 1)
                col.standardized()
            else
                col
        }

fun toXsAndY(cols: List<List<Double>>): Pair<MutableList<List<Double>>, List<Double>> {
    val y = cols[cols.lastIndex]
    val xs = cols.toMutableList()
    xs.removeAt(cols.lastIndex)
    return Pair(xs, y)
}

fun addFirstColumnsOfOnes(cols: MutableList<List<Double>>) {
    val ones = cols.first().map { 1.0 }
    cols.add(0, ones)
}

//Датасет — Facebook Comment Volume Dataset.
//Предсказать, сколько комментариев наберёт пост. Задача предполагает реализацию градиентного спуска и подсчёта метрик оценки качества модели.
// Можно использовать линейную алгебру, всякую другую математику готовую в либах.
//Этапы решения:
//— нормировка значений фичей;
//— кросс-валидация по пяти фолдам и обучение линейной регрессии;
//— подсчёт R^2 (коэффициента детерминации) и RMSE.
//Результаты можно оформить в виде следующей таблицы. T1,..T5 — фолды, E — среднее, STD — дисперсия,
// R^2/RMSE-test — значение соответствующей метрики на тестовой выборке, -train — на обучающей выборке,
// f0,..fn — значимость признаков (они же переменные, они же фичи).
//    T1,..T5 — фолды, E — среднее, STD — дисперсия, R^2/RMSE-test
fun main() {
    val filename = "D:\\Prog\\ml_hw_1\\src\\main\\resources\\Dataset\\Training\\Features_Variant_1.csv"
    val rows = readFile(filename)
    val cols = rows.cols()
    println("All data:")
    printSizeAndMeanAndStd(cols)

    val trainAndTest = rows.shuffled().chunked((rows.size * 0.80).toInt())
    val train = trainAndTest.first()
    val test = trainAndTest.last()

    val folds = train.shuffledAndChunked()
    for ((i, fold) in folds.withIndex()) {
        println("Fold #$i:")
        printSizeAndMeanAndStd(fold.cols())
    }

    val ws = mutableListOf<List<Double>>()

    for (k in 0 until 5) {
        println("k: $k")

        val kTest = folds[k]
        val mutableFolds = folds.toMutableList()
        mutableFolds.removeAt(k)
        val kTrain = mutableFolds.flatten().shuffled()

        val trainStandardizedCols = standardizeOnlyXsAndReturnCols(kTrain.cols())
        val (xsTrain, yTrain) = toXsAndY(trainStandardizedCols)
        addFirstColumnsOfOnes(xsTrain)
        val rowsTrain = xsTrain.transposed()

        val rate = 0.1
        val w = `stohastic gradient descent`(rowsTrain, yTrain, rate, 40000, stopOnSmallR2Changes = true)

        val kTestStandardizedCols = standardizeOnlyXsAndReturnCols(kTest.cols())
        val (xsKTest, yKTest) = toXsAndY(kTestStandardizedCols)
        addFirstColumnsOfOnes(xsKTest)
        val rowsKTest = xsKTest.transposed()

        val tss = `total sum of squares`(yKTest)
        val yPred = rowsKTest.map { x -> x.reduceIndexed { j, acc, d -> acc + d * w[j] } }
        println("k: $k Test:")
        println("MSE: ${mse(rowsKTest, yKTest, w)}. RMSE: ${rmse(rowsKTest, yKTest, w)}. R2: ${`coefficient of determination`(yKTest, yPred, tss)}")

        println("weights:")
        for ((j, d) in w.withIndex()) {
            println("F$j: $d")
        }

        ws.add(w)
    }

    val meanW = mutableListOf<Double>()
    for (j in ws.first().indices) {
        var wJ = 0.0
        var n = ws.size
        for (w in ws) {
            wJ += w[j]
        }

        wJ /= n
        meanW.add(wJ)
    }

    println("mean weights:")
    for ((j, d) in meanW.withIndex()) {
        println("F$j: $d")
    }

    val testStandardizedCols = standardizeOnlyXsAndReturnCols(test.cols())
    val (xsTest, yTest) = toXsAndY(testStandardizedCols)
    addFirstColumnsOfOnes(xsTest)
    val rowsTest = xsTest.transposed()

    val tss = `total sum of squares`(yTest)
    val yPred = rowsTest.map { x -> x.reduceIndexed { j, acc, d -> acc + d * meanW[j] } }
    println("Test:")
    println("MSE: ${mse(rowsTest, yTest, meanW)}. RMSE: ${rmse(rowsTest, yTest, meanW)}. R2: ${`coefficient of determination`(yTest, yPred, tss)}")
}