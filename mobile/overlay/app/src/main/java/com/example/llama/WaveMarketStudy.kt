package com.example.llama

import android.content.Context
import java.util.Locale

/**
 * All native Binance timeframe data is read at ONE historical H1 cutoff.
 * AO/AC, visible displaced Alligator and 2-right-bar confirmed fractals
 * are observable research evidence, NOT an authenticated Elliott wave count.
 */
internal object WaveMarketStudy {
    private const val VENUE = WaveMarketResearch.VENUE
    private const val MIN_BARS = 45

    data class Snapshot(
        val tf: String, val bars: Int, val lastCloseMs: Long,
        val close: Double, val ao: Double?, val ac: Double?,
        val jaw: Double?, val teeth: Double?, val lips: Double?,
        val momentum: String, val confirmedHighs: Int, val confirmedLows: Int,
        val nearestConfirmedHigh: Double?, val nearestConfirmedLow: Double?
    )

    private fun sma(values: DoubleArray, period: Int, at: Int): Double? {
        if (at < period - 1) return null
        var total = 0.0
        for (i in (at - period + 1)..at) total += values[i]
        return total / period
    }

    private fun smma(values: DoubleArray, period: Int): DoubleArray {
        val result = DoubleArray(values.size) { Double.NaN }
        if (values.size < period) return result
        var value = 0.0
        for (i in 0 until period) value += values[i]
        value /= period
        result[period - 1] = value
        for (i in period until values.size) {
            value = (value * (period - 1) + values[i]) / period
            result[i] = value
        }
        return result
    }

    fun snapshot(tf: String, bars: List<WaveMarketStore.Bar>): Snapshot? {
        if (bars.size < MIN_BARS) return null
        val highs = DoubleArray(bars.size) { bars[it].high.toDouble() }
        val lows = DoubleArray(bars.size) { bars[it].low.toDouble() }
        val closes = DoubleArray(bars.size) { bars[it].close.toDouble() }
        val median = DoubleArray(bars.size) { (highs[it] + lows[it]) / 2.0 }
        require((highs + lows + closes).all { it.isFinite() })
        val n = bars.size - 1
        val aoSeries = DoubleArray(bars.size) { Double.NaN }
        for (i in 33..n) {
            aoSeries[i] = (sma(median, 5, i) ?: 0.0) -
                (sma(median, 34, i) ?: 0.0)
        }
        val ao = if (aoSeries[n].isFinite()) aoSeries[n] else null
        val ac = if (ao != null && n >= 37) {
            val lastFive = (n - 4..n).map { aoSeries[it] }
            if (lastFive.all { it.isFinite() }) ao - lastFive.average() else null
        } else null
        val jawArr = smma(median, 13)
        val teethArr = smma(median, 8)
        val lipsArr = smma(median, 5)
        fun visible(series: DoubleArray, shift: Int): Double? {
            val idx = n - shift
            return if (idx >= 0 && series[idx].isFinite()) series[idx] else null
        }
        val jaw = visible(jawArr, 8)
        val teeth = visible(teethArr, 5)
        val lips = visible(lipsArr, 3)
        val highsConfirmed = mutableListOf<Double>()
        val lowsConfirmed = mutableListOf<Double>()
        for (i in 2 until bars.size - 2) {
            if (highs[i] > highs[i-1] && highs[i] > highs[i-2] &&
                highs[i] > highs[i+1] && highs[i] > highs[i+2])
                highsConfirmed.add(highs[i])
            if (lows[i] < lows[i-1] && lows[i] < lows[i-2] &&
                lows[i] < lows[i+1] && lows[i] < lows[i+2])
                lowsConfirmed.add(lows[i])
        }
        val phase = when {
            ao == null -> "AO_UNAVAILABLE"
            ao > 0 && aoSeries[n-1] < ao -> "AO_POSITIVE_RISING"
            ao > 0 -> "AO_POSITIVE_FALLING"
            ao < 0 && aoSeries[n-1] > ao -> "AO_NEGATIVE_FALLING"
            ao < 0 -> "AO_NEGATIVE_RISING"
            else -> "AO_NEAR_ZERO"
        }
        return Snapshot(tf, bars.size, bars.last().closeMs, closes[n],
            ao, ac, jaw, teeth, lips, phase, highsConfirmed.size,
            lowsConfirmed.size, highsConfirmed.lastOrNull(), lowsConfirmed.lastOrNull())
    }

    private fun Double?.display(): String =
        if (this == null || !isFinite()) "?" else
            String.format(Locale.US, "%.5f", this)

    fun readOnlyPassport(context: Context, symbol: String): String {
        require(symbol in WaveMarketResearch.SYMBOLS)
        return WaveMarketStore(context).use { cache ->
            val anchor = cache.progress(VENUE, symbol, "1h").lastClose
            if (anchor == null) return@use (
                "Binance USD-M $symbol: закрытые H1 пока не загружены. " +
                "Нет рыночных данных; никакая волна не подтверждена.")
            val lines = mutableListOf<String>()
            lines.add("Binance PUBLIC USD-M $symbol. Единый H1 UTC cutoff=$anchor.")
            lines.add("Это наблюдения, а не экспертная волновая разметка. " +
                "WM1/WM2/WM3, степень волн, входы и выходы не подтверждены. " +
                "Любые биржевые ордера запрещены.")
            var observed = 0
            for (tf in WaveMarketResearch.INTERVALS) {
                val bars = cache.recent(VENUE, symbol, tf, anchor, 700)
                val snap = snapshot(tf, bars)
                if (snap == null) {
                    lines.add(tf + ": недостаточно закрытой истории (" +
                        bars.size + "/$MIN_BARS). Wave=UNKNOWN.")
                    continue
                }
                observed++
                lines.add(
                    tf + ": bars=" + snap.bars +
                    ", close=" + snap.close.display() +
                    ", AO=" + snap.ao.display() +
                    ", AC=" + snap.ac.display() +
                    ", jaw=" + snap.jaw.display() +
                    ", teeth=" + snap.teeth.display() +
                    ", lips=" + snap.lips.display() +
                    ", phase=" + snap.momentum +
                    ", confirmed_hi=" + snap.confirmedHighs +
                    ", confirmed_lo=" + snap.confirmedLows +
                    ", last_hi=" + snap.nearestConfirmedHigh.display() +
                    ", last_lo=" + snap.nearestConfirmedLow.display() +
                    ", Elliott_degree=UNVERIFIED"
                )
            }
            lines.add("Индикаторные TFs=" + observed + "/15. Длина волны 100–140 " +
                "баров ориентировочна; вложенность W3/W5 не подтверждена. " +
                "Минутные свечи после H1 cutoff не использованы.")
            lines.joinToString("\n").take(5_000)
        }
    }
}
