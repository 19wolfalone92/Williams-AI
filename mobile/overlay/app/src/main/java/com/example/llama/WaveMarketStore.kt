package com.example.llama

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.math.BigDecimal

/**
 * Independent Williams-AI private research cache. Never contains account keys,
 * orders, positions, trade authorization, or a bridge to the Williams Bot.
 */
internal class WaveMarketStore(context: Context) : SQLiteOpenHelper(
    context.applicationContext, "williams_ai_public_futures_waves.db", null, 1
) {
    data class Bar(
        val openMs: Long, val closeMs: Long,
        val open: String, val high: String, val low: String, val close: String,
        val volume: String, val quoteVolume: String, val trades: Long,
        val takerBase: String, val takerQuote: String,
    )
    data class Progress(val count: Long, val firstOpen: Long?, val lastOpen: Long?,
                        val lastClose: Long?)

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""CREATE TABLE IF NOT EXISTS candles(
            venue TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            open_ms INTEGER NOT NULL, close_ms INTEGER NOT NULL,
            o TEXT NOT NULL, h TEXT NOT NULL, l TEXT NOT NULL, c TEXT NOT NULL,
            base_volume TEXT NOT NULL, quote_volume TEXT NOT NULL,
            trade_count INTEGER NOT NULL, taker_base TEXT NOT NULL, taker_quote TEXT NOT NULL,
            PRIMARY KEY(venue,symbol,timeframe,open_ms)
        )""".trimIndent())
        db.execSQL("""CREATE TABLE IF NOT EXISTS stream_meta(
            venue TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            first_requested_ms INTEGER NOT NULL,
            last_checked_server_ms INTEGER NOT NULL,
            price_tick TEXT NOT NULL,
            PRIMARY KEY(venue,symbol,timeframe)
        )""".trimIndent())
    }
    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Fail closed: never drop research or silently migrate incompatible timestamps.
        error("Wave archive migration must be explicitly reviewed")
    }

    private fun canonical(s: String, allowZero: Boolean = false): String {
        val number = BigDecimal(s)
        require(number.signum() >= if (allowZero) 0 else 1) { "Неверная биржевая цена/объём" }
        return number.stripTrailingZeros().toPlainString()
    }

    /** Returns canonical exchange strings; numeric validation is in collector too. */
    fun save(venue: String, symbol: String, tf: String, since: Long,
             serverMs: Long, tick: String, bars: List<Bar>) {
        require(venue == "PUBLIC_USDM" || venue == "DEMO_USDM")
        require(WaveMarketResearch.INTERVALS.contains(tf))
        require(symbol.matches(Regex("[A-Z0-9]{2,24}USDT")))
        val db = writableDatabase
        db.beginTransaction()
        try {
            val existingFirst = db.rawQuery(
                "SELECT first_requested_ms FROM stream_meta WHERE venue=? AND symbol=? AND timeframe=?",
                arrayOf(venue, symbol, tf)
            ).use { c -> if (c.moveToFirst()) c.getLong(0) else null }
            require(existingFirst == null || existingFirst == since) {
                "Начальная дата истории изменилась. Нужен отдельный архив."
            }
            if (existingFirst == null) {
                db.execSQL("INSERT INTO stream_meta VALUES (?,?,?,?,?,?)",
                    arrayOf(venue, symbol, tf, since, serverMs, tick))
            } else {
                db.execSQL("""UPDATE stream_meta SET last_checked_server_ms=?, price_tick=?
                    WHERE venue=? AND symbol=? AND timeframe=?""".trimIndent(),
                    arrayOf(serverMs, tick, venue, symbol, tf))
            }
            var previous = lastOpenInTransaction(db, venue, symbol, tf)
            for (bar in bars) {
                val existing = db.rawQuery(
                    """SELECT close_ms,o,h,l,c,base_volume,quote_volume,trade_count,taker_base,taker_quote
                       FROM candles WHERE venue=? AND symbol=? AND timeframe=? AND open_ms=?""".trimIndent(),
                    arrayOf(venue, symbol, tf, bar.openMs.toString())
                ).use { c ->
                    if (!c.moveToFirst()) null else listOf(
                        c.getLong(0).toString(), c.getString(1), c.getString(2),
                        c.getString(3), c.getString(4), c.getString(5),
                        c.getString(6), c.getLong(7).toString(),
                        c.getString(8), c.getString(9))
                }
                val fields = listOf(
                    bar.closeMs.toString(), canonical(bar.open), canonical(bar.high),
                    canonical(bar.low), canonical(bar.close),
                    canonical(bar.volume, true), canonical(bar.quoteVolume, true),
                    bar.trades.toString(), canonical(bar.takerBase, true),
                    canonical(bar.takerQuote, true)
                )
                if (existing != null) {
                    require(existing == fields) {
                        "Binance изменил ранее закрытую свечу $symbol/$tf — архив не обновлён"
                    }
                    continue
                }
                if (previous != null) {
                    require(bar.openMs == WaveMarketResearch.nextOpen(previous, tf)) {
                        "Пропуск/неупорядоченность свечей $symbol/$tf"
                    }
                }
                db.execSQL("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    arrayOf(venue, symbol, tf, bar.openMs, bar.closeMs,
                        fields[1], fields[2], fields[3], fields[4],
                        fields[5], fields[6], bar.trades, fields[8], fields[9]))
                previous = bar.openMs
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    private fun lastOpenInTransaction(db: SQLiteDatabase, venue: String,
                                      symbol: String, tf: String): Long? =
        db.rawQuery(
            "SELECT MAX(open_ms) FROM candles WHERE venue=? AND symbol=? AND timeframe=?",
            arrayOf(venue, symbol, tf)
        ).use { c ->
            if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else null
        }

    fun progress(venue: String, symbol: String, tf: String): Progress =
        readableDatabase.rawQuery(
            """SELECT COUNT(*),MIN(open_ms),MAX(open_ms),MAX(close_ms)
            FROM candles WHERE venue=? AND symbol=? AND timeframe=?""".trimIndent(),
            arrayOf(venue, symbol, tf)
        ).use { c ->
            check(c.moveToFirst())
            Progress(c.getLong(0),
                if (c.isNull(1)) null else c.getLong(1),
                if (c.isNull(2)) null else c.getLong(2),
                if (c.isNull(3)) null else c.getLong(3))
        }

    fun recent(venue: String, symbol: String, tf: String, asOf: Long, limit: Int): List<Bar> {
        require(limit in 1..2500 && asOf > 0)
        val descending = mutableListOf<Bar>()
        readableDatabase.rawQuery(
            """SELECT open_ms,close_ms,o,h,l,c,base_volume,quote_volume,trade_count,
            taker_base,taker_quote FROM candles WHERE venue=? AND symbol=? AND timeframe=?
            AND close_ms<=? ORDER BY open_ms DESC LIMIT ?""".trimIndent(),
            arrayOf(venue, symbol, tf, asOf.toString(), limit.toString())
        ).use { c ->
            while (c.moveToNext()) {
                descending.add(Bar(
                    c.getLong(0), c.getLong(1), c.getString(2),
                    c.getString(3), c.getString(4), c.getString(5),
                    c.getString(6), c.getString(7), c.getLong(8),
                    c.getString(9), c.getString(10)))
            }
        }
        val chronological = descending.reversed()
        for (i in 1 until chronological.size) {
            require(chronological[i].openMs ==
                WaveMarketResearch.nextOpen(chronological[i - 1].openMs, tf)) {
                "Найдены пропущенные свечи $tf. Анализ запрещён."
            }
        }
        return chronological
    }
}
