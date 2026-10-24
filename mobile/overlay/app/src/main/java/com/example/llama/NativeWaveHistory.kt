package com.example.llama

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.os.StatFs
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.math.BigDecimal
import java.net.HttpURLConnection
import java.net.URL
import java.time.Instant
import java.time.ZoneOffset
import java.time.ZonedDateTime
import java.time.temporal.TemporalAdjusters
import java.time.DayOfWeek
import java.util.Locale

/**
 * Public Binance USD-M FUTURES candles in app-private SQLite. NEVER credentials or orders.
 * Intentional limits: a user-initiated, bounded foreground batch, not a perpetual
 * Android service or a claim that ALL historical listings fit on the device.
 */
internal object NativeWaveConfig {
    const val VENUE = "PUBLIC_USDM"
    // One private SQLite/source-writer at a time across foreground and the
    // separately opt-in OS-persisted periodic research job.
    val marketSyncMutex = Any()
    private const val PINNED_HOST = "fapi.binance.com"
    const val ONE_MINUTE = 60_000L
    const val ONE_DAY = 86_400_000L
    const val FINALITY_MS = 2_000L
    const val MAX_PAGES_PER_INTERVAL_PER_TAP = 2
    const val PAGE_SIZE = 500
    const val MIN_FREE_STORAGE_BYTES = 100L * 1024 * 1024
    val symbols = listOf("ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","BTCUSDT",
        "DOGEUSDT","DOTUSDT","ETHUSDT","LINKUSDT","LTCUSDT",
        "NEARUSDT","SOLUSDT","SUIUSDT","TRXUSDT","XRPUSDT")
    val intervals = linkedMapOf(
        "1m" to 60_000L, "3m" to 180_000L, "5m" to 300_000L,
        "15m" to 900_000L, "30m" to 1_800_000L, "1h" to 3_600_000L,
        "2h" to 7_200_000L, "4h" to 14_400_000L, "6h" to 21_600_000L,
        "8h" to 28_800_000L, "12h" to 43_200_000L,
        "1d" to 86_400_000L, "3d" to 259_200_000L,
        "1w" to 604_800_000L, "1M" to -1L,
    )

    fun nextOpen(open: Long, tf: String): Long {
        require(intervals.containsKey(tf)) { "Неизвестный ТФ" }
        if (tf == "1M") {
            return ZonedDateTime.ofInstant(Instant.ofEpochMilli(open), ZoneOffset.UTC)
                .withDayOfMonth(1).toLocalDate().plusMonths(1)
                .atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli()
        }
        return Math.addExact(open, intervals.getValue(tf))
    }
    fun floorOpen(timestamp: Long, tf: String): Long {
        require(timestamp >= 0 && intervals.containsKey(tf)) { "Некорректное время/ТФ" }
        val instant = ZonedDateTime.ofInstant(Instant.ofEpochMilli(timestamp),ZoneOffset.UTC)
        if (tf == "1M") return instant.withDayOfMonth(1)
            .toLocalDate().atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli()
        if (tf == "1w") return instant.toLocalDate()
            .with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY))
            .atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli()
        if (tf == "3d") {
            // Binance actual 3d kline starts at UTC epoch + one day.
            val anchor = ONE_DAY
            return Math.floorDiv(timestamp-anchor, intervals.getValue(tf)) *
                intervals.getValue(tf) + anchor
        }
        val period = intervals.getValue(tf)
        return Math.floorDiv(timestamp,period)*period
    }
    fun verifyPrivatePublicUrl(url: URL) {
        require(url.protocol=="https" && url.host==PINNED_HOST &&
            url.port==-1 && url.path in setOf(
                "/fapi/v1/time","/fapi/v1/exchangeInfo","/fapi/v1/klines"
            )) { "Запрещённый источник графиков" }
    }
}

internal data class NativeCandle(val openTime:Long,val closeTime:Long,
    val open:String,val high:String,val low:String,val close:String,
    val baseVolume:String,val quoteVolume:String,val trades:Long,
    val takerBase:String,val takerQuote:String) {
    fun fingerprint() = listOf(closeTime,open,high,low,close,baseVolume,
        quoteVolume,trades,takerBase,takerQuote).joinToString("|")
}

internal object NativeWaveMarketApi {
    private fun get(path:String,query:String=""):String {
        val url=URL("https://fapi.binance.com"+path+
            (if (query.isEmpty()) "" else "?"+query))
        NativeWaveConfig.verifyPrivatePublicUrl(url)
        val conn = url.openConnection() as HttpURLConnection
        conn.instanceFollowRedirects=false
        conn.requestMethod="GET"
        conn.connectTimeout=10_000
        conn.readTimeout=15_000
        conn.setRequestProperty("Accept","application/json")
        conn.setRequestProperty("User-Agent","WilliamsAI-Android-PUBLIC-FUTURES-READONLY/1")
        val budget=NativeFuturesMetadata.responseBudget(path)
        try {
            val response=conn.responseCode
            check(response==200) { "BINANCE_PUBLIC_HTTP_${response}" }
            val received=conn.inputStream.use { input ->
                BufferedInputStream(input).use { buffered ->
                    val dest=java.io.ByteArrayOutputStream()
                    val block=ByteArray(8192)
                    while(true) {
                        val count=buffered.read(block)
                        if(count<0) break
                        check(dest.size()+count<=budget) {
                            "Ответ Binance превышает лимит"
                        }
                        dest.write(block,0,count)
                    }
                    dest.toString("UTF-8")
                }
            }
            return received
        } finally {
            conn.disconnect()
        }
    }

    fun serverTime():Long {
        val remote=JSONObject(get("/fapi/v1/time")).getLong("serverTime")
        check(kotlin.math.abs(remote-System.currentTimeMillis())<=5*60_000L) {
            "Системное время не синхронизировано с Binance"
        }
        return remote
    }
    fun symbolTick(symbol:String):String {
        require(symbol in NativeWaveConfig.symbols)
        val objectData=JSONObject(get("/fapi/v1/exchangeInfo"))
        val symbols=objectData.getJSONArray("symbols")
        val catalog=(0 until symbols.length()).map { i ->
            val row=symbols.getJSONObject(i)
            row.getString("symbol") to row.getString("status")
        }
        val pair=symbols.getJSONObject(NativeFuturesMetadata.contractIndex(symbol,catalog))
        val filters=pair.getJSONArray("filters")
        var tick:String?=null
        for (i in 0 until filters.length()) {
            val f=filters.getJSONObject(i)
            if (f.getString("filterType")=="PRICE_FILTER") {
                check(tick==null) { "Повторяющийся PRICE_FILTER биржи" }
                tick=positiveDecimal(f.getString("tickSize"))
            }
        }
        return tick ?: error("Не найден PRICE_FILTER биржи")
    }
    private fun positiveDecimal(s:String,zero:Boolean=false):String {
        check(s.length in 1..60) { "Число слишком длинное" }
        val d=BigDecimal(s)
        check(if(zero) d.signum()>=0 else d.signum()>0) { "Недопустимая величина" }
        return d.stripTrailingZeros().toPlainString()
    }
    fun candles(symbol:String,tf:String,start:Long,serverMs:Long):List<NativeCandle> {
        require(symbol in NativeWaveConfig.symbols && tf in NativeWaveConfig.intervals)
        require(start>=0 && serverMs>0)
        val query="symbol="+symbol+"&interval="+tf+"&startTime="+start+
            "&endTime="+(serverMs-NativeWaveConfig.FINALITY_MS)+
            "&limit="+NativeWaveConfig.PAGE_SIZE
        val arr=JSONArray(get("/fapi/v1/klines",query))
        check(arr.length()<=NativeWaveConfig.PAGE_SIZE) { "Некорректная страница" }
        val rows=ArrayList<NativeCandle>(arr.length())
        var prior:Long?=null
        for (i in 0 until arr.length()) {
            val v=arr.getJSONArray(i)
            require(v.length()>=11) { "Недостаточно полей свечи" }
            val op=v.getLong(0); val end=v.getLong(6)
            check(op>=start && op==NativeWaveConfig.floorOpen(op,tf) &&
                end==NativeWaveConfig.nextOpen(op,tf)-1L) { "Нарушен календарь Binance" }
            prior?.let {older->
                check(op==NativeWaveConfig.nextOpen(older,tf)) {
                    "Пропущена или дублирована свеча"
                }
            }
            prior=op
            if(end>=serverMs-NativeWaveConfig.FINALITY_MS) break
            val o=positiveDecimal(v.getString(1))
            val h=positiveDecimal(v.getString(2))
            val l=positiveDecimal(v.getString(3))
            val c=positiveDecimal(v.getString(4))
            check(BigDecimal(l)<=BigDecimal(o) && BigDecimal(l)<=BigDecimal(c) &&
                BigDecimal(h)>=BigDecimal(o) && BigDecimal(h)>=BigDecimal(c)) {
                "Повреждённые OHLC"
            }
            val volume=positiveDecimal(v.getString(5),true)
            val quote=positiveDecimal(v.getString(7),true)
            val trades=v.getLong(8)
            check(trades>=0) { "Некорректное число сделок" }
            val takerBase=positiveDecimal(v.getString(9),true)
            val takerQuote=positiveDecimal(v.getString(10),true)
            check(BigDecimal(takerBase)<=BigDecimal(volume) &&
                BigDecimal(takerQuote)<=BigDecimal(quote)) {
                "Taker-volume превышает общий объём"
            }
            rows+=NativeCandle(op,end,o,h,l,c,volume,quote,trades,takerBase,takerQuote)
        }
        return rows
    }
}

internal class NativeWaveHistoryDb(ctx:Context):SQLiteOpenHelper(
    ctx.applicationContext,"williams_ai_public_usdm_wave_v1.sqlite3",null,1) {
    override fun onCreate(db:SQLiteDatabase) {
        db.execSQL("""CREATE TABLE wave_candles(
            venue TEXT NOT NULL, symbol TEXT NOT NULL, tf TEXT NOT NULL,
            open_ms INTEGER NOT NULL, close_ms INTEGER NOT NULL,
            op TEXT NOT NULL, hi TEXT NOT NULL, lo TEXT NOT NULL, cl TEXT NOT NULL,
            vol TEXT NOT NULL, quote_vol TEXT NOT NULL, trades INTEGER NOT NULL,
            taker_base TEXT NOT NULL, taker_quote TEXT NOT NULL,
            PRIMARY KEY (venue,symbol,tf,open_ms)
        )""".trimIndent())
        db.execSQL("""CREATE TABLE wave_sources(
            venue TEXT NOT NULL,symbol TEXT NOT NULL,tf TEXT NOT NULL,
            tick_size TEXT NOT NULL,server_ms INTEGER NOT NULL,
            PRIMARY KEY (venue,symbol,tf)
        )""".trimIndent())
        db.execSQL("""CREATE TABLE wave_notes(
            venue TEXT NOT NULL,symbol TEXT NOT NULL,tf TEXT NOT NULL,
            algorithm_revision TEXT NOT NULL,cutoff_close_ms INTEGER NOT NULL,
            observed_json TEXT NOT NULL,
            PRIMARY KEY(venue,symbol,tf,algorithm_revision,cutoff_close_ms)
        )""".trimIndent())
    }
    override fun onUpgrade(db:SQLiteDatabase,oldVersion:Int,newVersion:Int) {
        error("Изменение схемы требует явной совместимой миграции")
    }
    fun lastOpen(symbol:String,tf:String):Long? =
        readableDatabase.rawQuery(
            "SELECT MAX(open_ms) FROM wave_candles WHERE venue=? AND symbol=? AND tf=?",
            arrayOf(NativeWaveConfig.VENUE,symbol,tf)).use { cur ->
            if(cur.moveToFirst() && !cur.isNull(0)) cur.getLong(0) else null
        }

    fun cachedCount(symbol:String,tf:String):Int =
        readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM wave_candles WHERE venue=? AND symbol=? AND tf=?",
            arrayOf(NativeWaveConfig.VENUE,symbol,tf)).use {
            if(it.moveToFirst()) it.getInt(0) else 0
        }

    fun savePage(symbol:String,tf:String,tick:String,server:Long,
                 rows:List<NativeCandle>):Int {
        require(symbol in NativeWaveConfig.symbols && tf in NativeWaveConfig.intervals)
        val db=writableDatabase
        db.beginTransaction()
        try {
            var previous=lastOpen(symbol,tf)
            var inserted=0
            rows.forEach { bar ->
                val found=db.rawQuery(
                    "SELECT close_ms,op,hi,lo,cl,vol,quote_vol,trades,taker_base,taker_quote "+
                        "FROM wave_candles WHERE venue=? AND symbol=? AND tf=? AND open_ms=?",
                    arrayOf(NativeWaveConfig.VENUE,symbol,tf,bar.openTime.toString())
                ).use { cur ->
                    if(!cur.moveToFirst()) null else
                        (0..9).joinToString("|") {
                            if(it==0 || it==7) cur.getLong(it).toString()
                            else cur.getString(it)
                        }
                }
                if(found!=null) {
                    check(found==bar.fingerprint()) { "Изменена ранее закрытая свеча Binance" }
                } else {
                    previous?.let { older ->
                        check(bar.openTime==NativeWaveConfig.nextOpen(older,tf)) {
                            "Обнаружен пропуск исторических свечей"
                        }
                    }
                    db.execSQL(
                        "INSERT INTO wave_candles VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        arrayOf(NativeWaveConfig.VENUE,symbol,tf,bar.openTime,bar.closeTime,
                            bar.open,bar.high,bar.low,bar.close,bar.baseVolume,
                            bar.quoteVolume,bar.trades,bar.takerBase,bar.takerQuote))
                    inserted++
                }
                previous=bar.openTime
            }
            db.execSQL("INSERT OR REPLACE INTO wave_sources VALUES(?,?,?,?,?)",
                arrayOf(NativeWaveConfig.VENUE,symbol,tf,tick,server))
            db.setTransactionSuccessful()
            return inserted
        } finally { db.endTransaction() }
    }

    data class Bar(val closeTime:Long,val open:Double,val high:Double,
                   val low:Double,val close:Double,val volume:Double)
    fun atMost(symbol:String,tf:String,asOf:Long,limit:Int=240):List<Bar> {
        require(symbol in NativeWaveConfig.symbols && tf in NativeWaveConfig.intervals)
        require(limit in 45..1500)
        val values=ArrayList<Bar>()
        readableDatabase.rawQuery(
            "SELECT close_ms,op,hi,lo,cl,vol FROM wave_candles "+
                "WHERE venue=? AND symbol=? AND tf=? AND close_ms<=? "+
                "ORDER BY open_ms DESC LIMIT ?",
            arrayOf(NativeWaveConfig.VENUE,symbol,tf,asOf.toString(),limit.toString())
        ).use { cur ->
            while(cur.moveToNext()) {
                values+=Bar(cur.getLong(0),cur.getDouble(1),cur.getDouble(2),
                    cur.getDouble(3),cur.getDouble(4),cur.getDouble(5))
            }
        }
        values.reverse()
        for(i in 1 until values.size) {
            // Actual native 1M calendar, 1w Monday, 3d +1d alignment.
            check(values[i].closeTime ==
                NativeWaveConfig.nextOpen(values[i-1].closeTime+1,tf)-1) {
                "История ${tf}: пробелы, анализ волн запрещён"
            }
        }
        return values
    }
    /** Aggregate proof that chronological archived checks were actually persisted. */
    fun historicalNoteCount(symbol:String,tf:String):Int {
        require(symbol in NativeWaveConfig.symbols && tf in NativeWaveConfig.intervals)
        return readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM wave_notes WHERE venue=? AND symbol=? AND tf=? "+
                "AND algorithm_revision=?",
            arrayOf(NativeWaveConfig.VENUE,symbol,tf,
                "ANDROID_NATIVE_15TF_WAVE_REVIEW_V1")
        ).use { if(it.moveToFirst())it.getInt(0) else 0 }
    }

    fun latestClosedHour():Long? {
        return readableDatabase.rawQuery(
            "SELECT MAX(close_ms) FROM wave_candles "+
                "WHERE venue=? AND tf='1h'",
            arrayOf(NativeWaveConfig.VENUE)).use {
                if(it.moveToFirst() && !it.isNull(0)) it.getLong(0) else null
            }
    }
}

/** User-initiated bounded sweep; restarts resume every stream from SQLite. */
internal class NativeWaveDownloader(private val ctx:Context) {
    fun onePass(cancelled:()->Boolean,onStatus:(String)->Unit,
                selectedSymbols:List<String> = NativeWaveConfig.symbols,
                pagesPerInterval:Int = NativeWaveConfig.MAX_PAGES_PER_INTERVAL_PER_TAP,
                maxDurationMs:Long = 24L*60_000L) {
        require(selectedSymbols.isNotEmpty() &&
            selectedSymbols.distinct().size==selectedSymbols.size &&
            selectedSymbols.all { it in NativeWaveConfig.symbols })
        require(pagesPerInterval in 1..NativeWaveConfig.MAX_PAGES_PER_INTERVAL_PER_TAP)
        require(maxDurationMs in 30_000L..24L*60_000L)
        val db=NativeWaveHistoryDb(ctx)
        try {
            val server=NativeWaveMarketApi.serverTime()
            var streams=0; var added=0; var errors=0; var notes=0
            var visited=0
            val totalPlanned=selectedSymbols.size*NativeWaveConfig.intervals.size
            val deadline=System.currentTimeMillis()+maxDurationMs
            for (symbol in selectedSymbols) {
                if(cancelled() || System.currentTimeMillis()>=deadline) break
                val tick=try { NativeWaveMarketApi.symbolTick(symbol) } catch(e:Exception) {
                    errors++
                    onStatus("${symbol}: контракт/доступ недоступен: ${e.message?.take(65)}")
                    if(e.message?.contains("HTTP_451")==true ||
                       e.message?.contains("HTTP_429")==true ||
                       e.message?.contains("время")==true) return
                    continue
                }
                for(tf in NativeWaveConfig.intervals.keys) {
                    if(cancelled() || System.currentTimeMillis()>=deadline) break
                    visited++
                    try {
                        val disk=StatFs(ctx.filesDir.absolutePath)
                        check(disk.availableBytes>NativeWaveConfig.MIN_FREE_STORAGE_BYTES) {
                            "Недостаточно свободного места; запись остановлена"
                        }
                        var cursor=db.lastOpen(symbol,tf) ?: 0L
                        for(page in 0 until pagesPerInterval) {
                            if(cancelled() || System.currentTimeMillis()>=deadline) break
                            val klines=NativeWaveMarketApi.candles(symbol,tf,cursor,server)
                            if(klines.isEmpty()) {
                                check(db.cachedCount(symbol,tf)>0) {
                                    "Не получены закрытые свечи $symbol/$tf"
                                }
                                break
                            }
                            added+=db.savePage(symbol,tf,tick,server,klines)
                            streams++
                            val next=NativeWaveConfig.nextOpen(klines.last().openTime,tf)
                            if(next<=cursor || next>=server-NativeWaveConfig.FINALITY_MS) break
                            cursor=next
                            if(klines.size<NativeWaveConfig.PAGE_SIZE) break
                            Thread.sleep(150L)
                        }
                        val progress=NativeWaveNotebook(db).review(
                            symbol,tf,maxCheckpoints=3,stride=120)
                        notes+=progress
                        onStatus("${symbol} ${tf}: ${db.cachedCount(symbol,tf)} свечей; "+
                            "заметок истории: ${db.historicalNoteCount(symbol,tf)}; "+
                            "новых свечей в цикле ${added}")
                    } catch(e:InterruptedException) { Thread.currentThread().interrupt(); return
                    } catch(e:Exception) {
                        errors++
                        onStatus("${symbol} ${tf}: ошибка ${e.message?.take(90)}; "+
                            "данные не выдумываются")
                        // Any 451, 429 or time/corruption issue: quit entire
                        // current pass; do not hammer Binance/loop another pair.
                        if(e.message?.contains("HTTP_451")==true ||
                           e.message?.contains("HTTP_429")==true ||
                           e.message?.contains("время")==true ||
                           e.message?.contains("Изменена")==true ||
                           e.message?.contains("Недостаточно свободного")==true) return
                    }
                }
            }
            val trulyComplete = visited==totalPlanned && errors==0 && !cancelled()
            onStatus(
                (if(trulyComplete) "Завершён ограниченный цикл: "
                 else "НЕПОЛНЫЙ цикл — не все TF проверены: ") +
                "проверено TF=" + visited + "/" + totalPlanned +
                ", свечей=" + added + ", страниц=" + streams +
                ", исторических заметок=" + notes + ", ошибок=" + errors +
                ". Глубина истории с листинга НЕ доказана. Сделок нет."
            )
        } finally { db.close() }
    }
}
