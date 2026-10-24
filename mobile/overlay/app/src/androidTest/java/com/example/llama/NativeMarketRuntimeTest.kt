package com.example.llama

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

private const val TEST_DB="williams_ai_runtime_test_only.sqlite3"
internal const val PROCESS_DB="williams_ai_process_test_only.sqlite3"
private const val HOUR=3_600_000L
private const val EPOCH=1_759_276_800_000L

internal fun testCandle(index:Int)=NativeCandle(
    EPOCH+index*HOUR,EPOCH+(index+1)*HOUR-1,
    "100","102","99","101","100","10000",12,"50","5000")

@RunWith(AndroidJUnit4::class)
class NativeMarketRuntimeTest {
    private val context:Context get()=InstrumentationRegistry.getInstrumentation().targetContext
    @Before fun cleanBefore() { context.deleteDatabase(TEST_DB) }
    @After fun cleanAfter() { context.deleteDatabase(TEST_DB) }

    @Test fun reopeningSqliteRetainsHistoryAndCutoffExcludesLaterCandles() {
        val bars=(0 until 120).map(::testCandle)
        val first=NativeWaveHistoryDb(context,TEST_DB)
        try { assertEquals(120,first.savePage("BTCUSDT","1h","0.01",bars.last().closeTime+5000,bars)) }
        finally { first.close() }
        val reopened=NativeWaveHistoryDb(context,TEST_DB)
        try {
            assertEquals(120,reopened.cachedCount("BTCUSDT","1h"))
            assertEquals(bars.last().openTime,reopened.lastOpen("BTCUSDT","1h"))
            val prefix=reopened.atMost("BTCUSDT","1h",bars[80].closeTime,120)
            assertEquals(81,prefix.size)
            assertTrue(prefix.all { it.closeTime<=bars[80].closeTime })
        } finally { reopened.close() }
    }

    @Test fun overlappedClosedPageIsIdempotentBeforeNewAppend() {
        val db=NativeWaveHistoryDb(context,TEST_DB)
        try {
            val bars=(0 until 60).map(::testCandle)
            db.savePage("BTCUSDT","1h","0.01",bars.last().closeTime+5000,bars)
            assertEquals(0,db.savePage("BTCUSDT","1h","0.01",bars.last().closeTime+5000,bars.takeLast(2)))
            assertEquals(1,db.savePage("BTCUSDT","1h","0.01",testCandle(60).closeTime+5000,
                listOf(bars.last(),testCandle(60))))
            assertEquals(61,db.cachedCount("BTCUSDT","1h"))
        } finally { db.close() }
    }

    @Test fun changedClosedCandleRollsBackWholeIncomingPage() {
        val db=NativeWaveHistoryDb(context,TEST_DB)
        try {
            db.savePage("BTCUSDT","1h","0.01",testCandle(2).closeTime+5000,(0..2).map(::testCandle))
            try {
                db.savePage("BTCUSDT","1h","0.01",testCandle(3).closeTime+5000,
                    listOf(testCandle(3),testCandle(1).copy(high="103")))
                fail("Changed closed history accepted")
            } catch (_:IllegalStateException) { }
            assertEquals(3,db.cachedCount("BTCUSDT","1h"))
            assertEquals(testCandle(2).openTime,db.lastOpen("BTCUSDT","1h"))
        } finally { db.close() }
    }

    @Test fun missingHourCannotBeCommitted() {
        val db=NativeWaveHistoryDb(context,TEST_DB)
        try {
            db.savePage("BTCUSDT","1h","0.01",testCandle(2).closeTime+5000,(0..2).map(::testCandle))
            try {
                db.savePage("BTCUSDT","1h","0.01",testCandle(4).closeTime+5000,listOf(testCandle(4)))
                fail("Gap accepted")
            } catch (_:IllegalStateException) { }
            assertEquals(3,db.cachedCount("BTCUSDT","1h"))
        } finally { db.close() }
    }
}

/** CI invokes these separately with an actual am force-stop between them. */
@RunWith(AndroidJUnit4::class)
class NativeMarketProcessCheckpointTest {
    private val context:Context get()=InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun seedBeforeProcessStop() {
        context.deleteDatabase(PROCESS_DB)
        val db=NativeWaveHistoryDb(context,PROCESS_DB)
        try {
            val bars=(0 until 60).map(::testCandle)
            assertEquals(60,db.savePage("ETHUSDT","1h","0.01",bars.last().closeTime+5000,bars))
        } finally { db.close() }
    }

    @Test fun verifyAfterProcessStop() {
        val db=NativeWaveHistoryDb(context,PROCESS_DB)
        try {
            assertEquals(60,db.cachedCount("ETHUSDT","1h"))
            assertEquals(testCandle(59).openTime,db.lastOpen("ETHUSDT","1h"))
            assertEquals(0,db.savePage("ETHUSDT","1h","0.01",testCandle(59).closeTime+5000,
                listOf(testCandle(59))))
        } finally { db.close();context.deleteDatabase(PROCESS_DB) }
    }
}
