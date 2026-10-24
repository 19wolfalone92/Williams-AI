package com.example.llama

import android.content.Context
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import java.math.BigDecimal
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.Calendar
import java.util.TimeZone
import javax.net.ssl.HttpsURLConnection
import kotlin.math.abs

/**
 * Real Binance Futures *public market data* only.
 * All network calls are GET to two fixed official USD-M hosts.
 * It cannot read API keys and has no order, account or trade-execution route.
 */
internal object WaveMarketResearch {
    const val VENUE = "PUBLIC_USDM"
    const val HOST = "fapi.binance.com"
    val SYMBOLS = listOf(
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
        "LTCUSDT", "BCHUSDT", "TRXUSDT", "NEARUSDT", "SUIUSDT"
    )
    val INTERVALS = listOf(
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
        "6h", "8h", "12h", "1d", "3d", "1w", "1M"
    )
    private const val HOUR = 3_600_000L
    private const val DAY = 86_400_000L
    private const val SAFETY = 2_000L
    private const val PAGE = 500

    private val periods = mapOf(
        "1m" to 60_000L, "3m" to 180_000L, "5m" to 300_000L,
        "15m" to 900_000L, "30m" to 1_800_000L, "1h" to HOUR,
        "2h" to 2 * HOUR, "4h" to 4 * HOUR, "6h" to 6 * HOUR,
        "8h" to 8 * HOUR, "12h" to 12 * HOUR,
        "1d" to DAY, "3d" to 3 * DAY, "1w" to 7 * DAY,
    )

    fun floorOpen(ms: Long, tf: String): Long {
        require(ms >= 0 && INTERVALS.contains(tf))
        if (tf == "1M") {
            val utc = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
            utc.timeInMillis = ms
            utc.set(Calendar.DAY_OF_MONTH, 1)
            utc.set(Calendar.HOUR_OF_DAY, 0)
            utc.set(Calendar.MINUTE, 0)
            utc.set(Calendar.SECOND, 0)
            utc.set(Calendar.MILLISECOND, 0)
            return utc.timeInMillis
        }
        if (tf == "1w") {
            val utc = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
            utc.timeInMillis = ms
            val daysSinceMonday = (utc.get(Calendar.DAY_OF_WEEK) + 5) % 7
            utc.add(Calendar.DAY_OF_MONTH, -daysSinceMonday)
            utc.set(Calendar.HOUR_OF_DAY, 0)
            utc.set(Calendar.MINUTE, 0)
            utc.set(Calendar.SECOND, 0)
            utc.set(Calendar.MILLISECOND, 0)
            return utc.timeInMillis
        }
        val width = periods[tf] ?: error("Недопустимый ТФ")
        // Empirically verified Binance USD-M 3d opens: 2026-10-23 UTC.
        val anchor = if (tf == "3d") DAY else 0L
        return Math.floorDiv(ms - anchor, width) * width + anchor
    }

    fun nextOpen(openMs: Long, tf: String): Long {
        require(INTERVALS.contains(tf))
        if (tf == "1M") {
            val cal = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
            cal.timeInMillis = openMs
            cal.add(Calendar.MONTH, 1)
            return cal.timeInMillis
        }
        return openMs + (periods[tf] ?: error("Недопустимый ТФ"))
    }

    private fun getPublic(endpoint: String, params: Map<String, String> = emptyMap()): String {
        require(endpoint in setOf("/fapi/v1/time", "/fapi/v1/exchangeInfo", "/fapi/v1/klines"))
        val url = URL("https://$HOST$endpoint" +
            (if (params.isEmpty()) "" else "?" +
                params.toSortedMap().entries.joinToString("&") {
                    Uri.encode(it.key) + "=" + Uri.encode(it.value)
                }))
        val conn = (url.openConnection() as HttpsURLConnection)
        conn.instanceFollowRedirects = false
        conn.connectTimeout = 8_000
        conn.readTimeout = 12_000
        conn.requestMethod = "GET"
        conn.setRequestProperty("Accept", "application/json")
        conn.setRequestProperty("User-Agent", "WilliamsAI-Android-Public-Waves/1")
        try {
            val code = conn.responseCode
            require(code == 200) { "Binance PUBLIC HTTP $code, заявки отключены" }
            val sizeCap = 2_500_000
            conn.inputStream.use { input ->
                val out = java.io.ByteArrayOutputStream()
                val bytes = ByteArray(16_384)
                while (true) {
                    val count = input.read(bytes)
                    if (count < 0) break
                    require(out.size() + count <= sizeCap) { "Binance ответ слишком большой" }
                    out.write(bytes, 0, count)
                }
                return out.toString(StandardCharsets.UTF_8.name())
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun dec(raw: String, allowZero: Boolean = false): BigDecimal {
        val value = BigDecimal(raw)
        require(value.signum() >= if (allowZero) 0 else 1) {
            "Биржа вернула некорректное значение свечи"
        }
        return value
    }

    fun validatedBar(row: JSONArray, tf: String, serverMs: Long): WaveMarketStore.Bar? {
        require(row.length() >= 11)
        val opened = row.getLong(0)
        val closed = row.getLong(6)
        require(opened >= 0 && floorOpen(opened, tf) == opened &&
            closed == nextOpen(opened, tf) - 1L) { "Ошибка UTC-границ свечи $tf" }
        if (closed >= serverMs - SAFETY) return null  // Never persist current candle.
        val o = dec(row.getString(1))
        val h = dec(row.getString(2))
        val l = dec(row.getString(3))
        val c = dec(row.getString(4))
        require(h >= l && h >= o && h >= c && l <= o && l <= c)
        val v = dec(row.getString(5), allowZero = true)
        val quote = dec(row.getString(7), allowZero = true)
        val trades = row.getLong(8)
        require(trades >= 0)
        val takerBase = dec(row.getString(9), allowZero = true)
        val takerQuote = dec(row.getString(10), allowZero = true)
        require(takerBase <= v && takerQuote <= quote)
        fun BigDecimal.norm() = stripTrailingZeros().toPlainString()
        return WaveMarketStore.Bar(opened, closed, o.norm(), h.norm(), l.norm(),
            c.norm(), v.norm(), quote.norm(), trades, takerBase.norm(), takerQuote.norm())
    }

    data class Result(
        val symbol: String, val interval: String, val cachedBars: Long,
        val newlyAdded: Long, val newestClosed: Long?, val state: String
    )

    /**
     * Executes at most ONE public Kline page for each of 15 timeframes per
     * bounded run. A stopped device resumes at persisted lastOpen; never
     * pretends that exchange listing-era history has been entirely fetched.
     */
    fun updateAll(
        context: Context, symbol: String, shouldStop: () -> Boolean = { false }
    ): List<Result> {
        require(symbol in SYMBOLS) { "Инструмент не входит в проверенный список" }
        val localNow = System.currentTimeMillis()
        val serverMs = JSONObject(getPublic("/fapi/v1/time")).getLong("serverTime")
        require(abs(serverMs - localNow) <= 300_000L) {
            "Ошибка времени Binance, исследование остановлено"
        }
        val exchangeInfo = JSONObject(getPublic("/fapi/v1/exchangeInfo",
            mapOf("symbol" to symbol)))
        val instruments = exchangeInfo.getJSONArray("symbols")
        require(instruments.length() == 1)
        val pair = instruments.getJSONObject(0)
        require(pair.getString("symbol") == symbol && pair.getString("status") == "TRADING") {
            "Фьючерсный контракт недоступен"
        }
        val filters = pair.getJSONArray("filters")
        var priceTick: String? = null
        for (i in 0 until filters.length()) {
            val filter = filters.getJSONObject(i)
            if (filter.getString("filterType") == "PRICE_FILTER")
                priceTick = dec(filter.getString("tickSize")).toPlainString()
        }
        val tick = requireNotNull(priceTick) { "Нет проверки биржевого шага цены" }
        val outputs = mutableListOf<Result>()
        WaveMarketStore(context).use { db ->
            for (tf in INTERVALS) {
                if (shouldStop()) break
                val stored = db.progress(VENUE, symbol, tf)
                // Start from earliest publicly available Binance candle when
                // empty; re-read one overlap after a restart for integrity.
                val cursor = stored.lastOpen ?: 0L
                val remote = JSONArray(getPublic("/fapi/v1/klines", mapOf(
                    "symbol" to symbol, "interval" to tf, "startTime" to cursor.toString(),
                    "endTime" to (serverMs - SAFETY).toString(), "limit" to PAGE.toString()
                )))
                require(remote.length() <= PAGE)
                val closed = mutableListOf<WaveMarketStore.Bar>()
                var prevOpen: Long? = null
                var sawPartial = false
                for (i in 0 until remote.length()) {
                    val bar = validatedBar(remote.getJSONArray(i), tf, serverMs)
                    if (bar == null) {
                        sawPartial = true
                        break
                    }
                    require(bar.openMs >= cursor) { "Binance переместил курсор истории назад" }
                    if (prevOpen != null) require(bar.openMs == nextOpen(prevOpen, tf)) {
                        "История Binance содержит пропуск $symbol/$tf"
                    }
                    prevOpen = bar.openMs
                    closed.add(bar)
                }
                require(closed.isNotEmpty() || stored.count > 0L) {
                    "Биржа не предоставила закрытых свечей $symbol/$tf"
                }
                db.save(VENUE, symbol, tf, 0L, serverMs, tick, closed)
                val after = db.progress(VENUE, symbol, tf)
                val latest = after.lastOpen
                val next = latest?.let { nextOpen(it, tf) }
                val currentFinalized = next != null &&
                    next >= floorOpen(serverMs - SAFETY, tf)
                val state = if (currentFinalized || sawPartial || remote.isEmpty()) {
                    "UP_TO_LATEST_AVAILABLE"  // NOT entire listing-era archive
                } else "ARCHIVE_BACKFILL_IN_PROGRESS"
                outputs.add(Result(symbol, tf, after.count,
                    after.count - stored.count, after.lastClose, state))
            }
        }
        return outputs
    }
}
