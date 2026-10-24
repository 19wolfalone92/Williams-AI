package com.example.llama

import android.content.Context
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlin.math.abs
import kotlin.math.max

/**
 * Compact source-bounded native market evidence for local GGUF dialogue.
 *
 * This is deterministic math over saved, CLOSED Binance PUBLIC_USDM candles.
 * No network in chat, no LLM-mutated cache, no source inference across venues,
 * and no claim that an unlabelled pivot is a verified W3/W4/W5.
 */
internal object NativeWaveStudy {
    private val atUtc=DateTimeFormatter.ofPattern("dd.MM HH:mm 'UTC'")
        .withZone(ZoneOffset.UTC)
    private fun date(t:Long):String=atUtc.format(Instant.ofEpochMilli(t))
    private fun sma(values:List<Double>,period:Int,i:Int):Double? {
        if(i+1<period) return null
        return (i-period+1..i).sumOf { values[it] } / period
    }
    private fun smma(values:List<Double>,period:Int,at:Int):Double? {
        if (at<period-1) return null
        var value=values.subList(0,period).average()
        for(i in period..at) value=(value*(period-1)+values[i])/period
        return value
    }
    private fun f(v:Double?):String {
        if(v==null || !v.isFinite()) return "НЕТ"
        return String.format(java.util.Locale.US,"%.5f",v)
    }
    private data class WaveFacts(
        val closedAt:Long,val bars:Int,val close:Double,val ao:Double?,
        val ac:Double?,val context:String,val lastPivot:String,
        val nearZeroAo:Boolean,val jaw:Double?,val teeth:Double?,val lips:Double?,
    )

    private fun facts(bars:List<NativeWaveHistoryDb.Bar>):WaveFacts? {
        if(bars.size<45) return null
        val med=bars.map{(it.high+it.low)/2.0}
        val osc=List(bars.size){i->
            val a=sma(med,5,i);val b=sma(med,34,i)
            if(a==null || b==null) null else a-b
        }
        val ao=osc.last()
        val last=bars.lastIndex
        val ac=if(last>=37 && osc.subList(last-4,last+1).all{it!=null})
            ao!!-osc.subList(last-4,last+1).map{it!!}.average() else null
        // Displaced Alligator values at graphical time t derive ONLY from
        // t-8/5/3, not from future t+shift values.
        val jaw=smma(med,13,last-8)
        val teeth=smma(med,8,last-5)
        val lips=smma(med,5,last-3)
        val price=bars.last().close
        val phase=when {
            jaw!=null && teeth!=null && lips!=null &&
                price>lips && lips>teeth && teeth>jaw -> "UP_CONTEXT"
            jaw!=null && teeth!=null && lips!=null &&
                price<lips && lips<teeth && teeth<jaw -> "DOWN_CONTEXT"
            else -> "UNRESOLVED"
        }
        // Current pivots confirm strictly after two completed bars to
        // their RIGHT. With a single last bar at index t, only t-2 is
        // eligible, not the current extreme and not a future price.
        var pivot="недостаточно подтверждённых экстремумов"
        for(i in bars.lastIndex-2 downTo max(2,bars.size-100)) {
            val h=bars[i].high
            val l=bars[i].low
            val high=(i-2..i+2).all{k->k==i || h>bars[k].high}
            val low=(i-2..i+2).all{k->k==i || l<bars[k].low}
            if(high!=low) {
                pivot=(if(high)"HIGH " else "LOW ")+date(bars[i].closeTime)+
                    " подтверждён через 2 закрытых бара"
                break
            }
        }
        val visibleAo=osc.takeLast(140).filterNotNull()
        val maxAbs=visibleAo.maxOfOrNull{abs(it)} ?:0.0
        val near=ao!=null && maxAbs>0 && abs(ao)<=maxAbs*0.15
        return WaveFacts(bars.last().closeTime,bars.size,price,ao,ac,phase,pivot,
            near,jaw,teeth,lips)
    }

    /**
     * Private, immutable historical note at a past closed candle; can be used
     * to review evidence seen THEN, not to back-label pivots using later data.
     */
    fun archiveNote(bars:List<NativeWaveHistoryDb.Bar>):String {
        val obs=facts(bars) ?: error("Требуется >=45 закрытых баров для заметки")
        return org.json.JSONObject().apply {
            put("algorithm_revision","ANDROID_NATIVE_15TF_WAVE_REVIEW_V1")
            put("cutoff_close_ms",obs.closedAt)
            put("available_closed_bars",obs.bars)
            put("price_close",f(obs.close))
            put("ao",f(obs.ao))
            put("ac",f(obs.ac))
            put("jaw_visible",f(obs.jaw))
            put("teeth_visible",f(obs.teeth))
            put("lips_visible",f(obs.lips))
            put("market_context",obs.context)
            put("recent_confirmed_pivot",obs.lastPivot)
            put("ao_near_zero_only_hypothesis",obs.nearZeroAo)
            put("elliott_wave_degree_verified",false)
            put("actual_trade_action",org.json.JSONObject.NULL)
            put("future_bars_used",false)
            put("weights_gguf_retrained",false)
        }.toString()
    }

    /** Only the symbols baked into the app are eligible, not remote model URLs. */
    fun symbolForQuestion(question:String):String? {
        val uppercase=question.uppercase(java.util.Locale.ROOT)
        val exact=NativeWaveConfig.symbols.firstOrNull{
            Regex("(?<![A-Z0-9])"+it+"(?![A-Z0-9])").containsMatchIn(uppercase)
        }
        if(exact!=null) return exact
        val thematic=Regex("(?iu)волн|график|индикатор|котиров|рынок|вильямс|трейд|wm[123]|ao|alligator|фрактал")
        return if(thematic.containsMatchIn(question)) "ETHUSDT" else null
    }
    fun prompt(ctx:Context,question:String):String {
        val symbol=symbolForQuestion(question) ?: return ""
        val db=NativeWaveHistoryDb(ctx)
        try {
            val h1=db.atMost(symbol,"1h",Long.MAX_VALUE,60)
            if(h1.isEmpty()) return "РЕАЛЬНЫХ СВЕЧЕЙ BINANCE ДЛЯ ${symbol} В ANDROID ЕЩЁ НЕТ. "+
                "Сначала запустите загрузку данных; не придумывайте график.\n"
            val asOf=h1.last().closeTime
            val stale=System.currentTimeMillis()-asOf>2*3_600_000L
            val out=StringBuilder(
                "ПУБЛИЧНЫЕ ДАННЫЕ BINANCE USD-M FUTURES. Только закрытые "+
                "свечи из приватной SQLite, НЕ DEMO и НЕ торговый счёт.\n"+
                "Инструмент: ${symbol}. Общий причинный срез: ${date(asOf)}. "+
                (if(stale)"ВНИМАНИЕ: данные устарели; никаких текущих цен и торговых сигналов.\n"
                 else "Это наблюдения, не биржевые исполнения.\n")+
                "15 независимых родных таймфреймов; степень волн не подтверждена.\n"
            )
            for(tf in NativeWaveConfig.intervals.keys) {
                val bars=db.atMost(symbol,tf,asOf,240)
                val read=facts(bars)
                if(read==null){
                    out.append("${tf}: INSUFFICIENT CLOSED CANDLES (${bars.size}/45).\n")
                } else {
                    // A still-available old monthly candle is valid historical
                    // context but cannot be mistaken for a new live bar.
                    out.append("${tf}: конец ${date(read.closedAt)}, "+
                        "закрытых ${read.bars}; C=${f(read.close)}, "+
                        "AO=${f(read.ao)}, AC=${f(read.ac)}, "+
                        "Alligator J/T/L=${f(read.jaw)}/${f(read.teeth)}/${f(read.lips)}, "+
                        "контекст=${read.context}; "+
                        (if(read.nearZeroAo)"AO близко к 0 (ГИПОТЕЗА коррекции); " else "")+
                        "pivot=${read.lastPivot}; W3/W4/W5 НЕ ДОКАЗАНЫ.\n")
                }
            }
            out.append("НЕ называй временную вложенность двух TF подтверждённой "+
                "волной Эллиотта. 100–140 баров — ориентир, не закон. "+
                "Нет цены входа, подтверждённого SL или права совершать ордера. "+
                "Данные в чате НЕ означают переобучение весов GGUF.\n")
            return out.toString().take(7200)
        } catch (e:Exception) {
            return "Верификация локальной Binance-истории не пройдена: "+
                e.javaClass.simpleName+". Не утверждай, что видел график.\n"
        } finally { db.close() }
    }
}
