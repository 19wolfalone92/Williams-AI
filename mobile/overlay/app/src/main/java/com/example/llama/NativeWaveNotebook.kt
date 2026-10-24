package com.example.llama

import android.database.sqlite.SQLiteDatabase

/**
 * History gets investigated left-to-right from first accessible candle.
 * Each archived cut sees only bars <= cut, then progress persists in SQLite.
 * "Study" here = observable wave/indicator research notes, never gradient
 * training or self-editing GGUF model weights and never live orders.
 */
internal class NativeWaveNotebook(private val db:NativeWaveHistoryDb) {
    private val revision="ANDROID_NATIVE_15TF_WAVE_REVIEW_V1"
    fun review(symbol:String,tf:String,maxCheckpoints:Int=3,stride:Int=120):Int {
        require(symbol in NativeWaveConfig.symbols &&
            tf in NativeWaveConfig.intervals &&
            maxCheckpoints in 1..50 && stride in 1..300)
        val sql=db.writableDatabase
        var saved=0
        for(check in 0 until maxCheckpoints) {
            val last=sql.rawQuery(
                "SELECT MAX(cutoff_close_ms) FROM wave_notes WHERE "+
                    "venue=? AND symbol=? AND tf=? AND algorithm_revision=?",
                arrayOf(NativeWaveConfig.VENUE,symbol,tf,revision)
            ).use { x->if(x.moveToFirst() && !x.isNull(0))x.getLong(0) else null }
            val selectedCursor=if(last==null) {
                sql.rawQuery(
                    "SELECT open_ms,close_ms FROM wave_candles WHERE "+
                        "venue=? AND symbol=? AND tf=? "+
                        "ORDER BY open_ms ASC LIMIT 1 OFFSET 44",
                    arrayOf(NativeWaveConfig.VENUE,symbol,tf))
            } else {
                sql.rawQuery(
                    "SELECT open_ms,close_ms FROM wave_candles WHERE "+
                        "venue=? AND symbol=? AND tf=? AND close_ms>? "+
                        "ORDER BY open_ms ASC LIMIT 1 OFFSET ${stride-1}",
                    arrayOf(NativeWaveConfig.VENUE,symbol,tf,last.toString()))
            }
            val row=selectedCursor.use { cursor->
                if(cursor.moveToFirst()) Pair(cursor.getLong(0),cursor.getLong(1))
                else null
            } ?: break
            // The SQL stream index is a closed candle. Fetch only its past:
            // no later bars, unconfirmed future pivots, or missing hours.
            val bars=db.atMost(symbol,tf,row.second,240)
            check(bars.size>=45 && bars.last().closeTime==row.second) {
                "Архивная история неполная, волновая проверка запрещена"
            }
            val note=NativeWaveStudy.archiveNote(bars)
            sql.beginTransaction()
            try {
                sql.execSQL(
                    "INSERT INTO wave_notes VALUES(?,?,?,?,?,?)",
                    arrayOf(NativeWaveConfig.VENUE,symbol,tf,revision,row.second,note))
                sql.setTransactionSuccessful()
            } finally {sql.endTransaction()}
            saved++
        }
        return saved
    }
}
