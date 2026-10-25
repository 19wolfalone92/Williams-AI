package com.example.llama

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import java.io.File

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

    /** Synthetic text fixtures; matching metadata is NOT PDF verification. */
    private fun withBooks(test: (File, File) -> Unit) {
        val books = File.createTempFile("ai-books-test-", ".sqlite3", context.cacheDir)
        val notes = File.createTempFile("ai-notes-test-", ".db", context.cacheDir)
        try {
            SQLiteDatabase.openOrCreateDatabase(books, null).use { db ->
                db.execSQL("CREATE TABLE volumes(book TEXT,sha256 TEXT,total_pages INTEGER," +
                    "extracted_pages INTEGER,missing_pages INTEGER)")
                db.execSQL("CREATE TABLE excerpts(book TEXT,pdf_page INTEGER,content TEXT)")
                // The minified target APK can remove kotlin.Triple when
                // only the separate test APK uses it. Use platform arrays.
                val ids = arrayOf("TC1", "NTD", "TC2", "COURSE")
                val hashes = arrayOf(
                    "36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c",
                    "855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4",
                    "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f",
                    "fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca")
                val counts = intArrayOf(140, 156, 224, 122)
                for (i in ids.indices) {
                    val id = ids[i]
                    val hash = hashes[i]
                    val count = counts[i]
                    db.execSQL("INSERT INTO volumes VALUES(?,?,?,?,?)",
                        arrayOf<Any>(id, hash, count, 1, count - 1))
                    db.execSQL("INSERT INTO excerpts VALUES(?,?,?)", arrayOf<Any>(id, 10,
                        "Фрактал и стоп проверяются по закрытым свечам. Учебный текст " + id + "."))
                }
            }
            test(books, notes)
        } finally { books.delete(); notes.delete() }
    }

    @Test fun targetedBookRetrievalDoesNotMixEditions() = withBooks { books, _ ->
        val selected = BookKnowledge.retrieve(books, "фрактал стоп", 3, "TC2")
        assertEquals(1, selected.size)
        assertEquals("TC2", selected.first().book)
        assertTrue(BookKnowledge.citationsSupported("[TC2, PDF с. 10]", selected))
        assertTrue(!BookKnowledge.citationsSupported("[TC2, PDF с. 131]", selected))
        assertTrue(!BookKnowledge.citationsSupported("[NTD, PDF с. 10]", selected))
    }

    @Test fun onlyFinalDeclaredRuleGapRequestsRereading() {
        assertEquals("EXIT", BookKnowledge.reviewTopic("[НУЖНА КНИГА: EXIT] Неясно правило выхода."))
        assertEquals(null, BookKnowledge.reviewTopic("[НУЖНЫ ДАННЫЕ] Нет закрытых H4."))
        assertEquals(null, BookKnowledge.reviewTopic("Нет сигнала; ожидаю ценовой триггер."))
        assertEquals(null, BookKnowledge.reviewTopic("<think>[НУЖНА КНИГА: ENTRY]</think>Ожидаю."))
        assertEquals(null, BookKnowledge.reviewTopic("<think>[НУЖНА КНИГА: ENTRY]"))
    }

    @Test fun learnedSourceNoteSurvivesReopenButNotModelOrOcrChange() = withBooks { books, notes ->
        val source = BookKnowledge.retrieve(books, "фрактал стоп", 1, "TC2").first()
        val p = StudyMemory.passage(books, source) ?: error("Fixture source not found")
        val raw = JSONObject().put("summary", "Фрактал и стоп требуют закрытых свечей.")
            .put("support", "Фрактал и стоп проверяются по закрытым свечам.").toString()
        StudyMemory.save(notes, p, "gguf-first", raw)
        val recalled = StudyMemory.retrieve(notes, "фрактал стоп", "gguf-first", books, "TC2")
        assertTrue(recalled.isNotEmpty())
        assertTrue(!recalled.contains("Фрактал и стоп требуют закрытых свечей."))
        assertEquals("", StudyMemory.retrieve(notes, "фрактал стоп", "gguf-second", books, "TC2"))
        assertEquals("", StudyMemory.retrieve(notes, "фрактал стоп", "gguf-first", books, "NTD"))
        SQLiteDatabase.openDatabase(books.absolutePath, null, SQLiteDatabase.OPEN_READWRITE).use { db ->
            db.execSQL("UPDATE excerpts SET content=? WHERE book='TC2'",
                arrayOf<Any>("Исправленная OCR версия: фрактал и стоп требуют иной проверки."))
        }
        assertEquals("", StudyMemory.retrieve(notes, "фрактал стоп", "gguf-first", books, "TC2"))
    }

    @Test fun inventedLearningSupportIsRejectedBeforePersistence() = withBooks { books, notes ->
        val source = BookKnowledge.retrieve(books, "фрактал стоп", 1, "TC2").first()
        val p = StudyMemory.passage(books, source) ?: error("Fixture source not found")
        try {
            StudyMemory.save(notes, p, "gguf-first", JSONObject()
                .put("summary", "Фрактал стоп.")
                .put("support", "This sentence was never in the supplied source.").toString())
            fail("Invented quote accepted")
        } catch (_: IllegalArgumentException) { }
        assertEquals("", StudyMemory.retrieve(notes, "фрактал стоп", "gguf-first", books, "TC2"))
    }

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
                    listOf(testCandle(3),testCandle(1).let { prior ->
                        // Android's minified production APK may remove
                        // Kotlin data-class copy$default; use its public
                        // constructor in the separately compiled test APK.
                        NativeCandle(
                            prior.openTime,prior.closeTime,
                            prior.open,"103",prior.low,prior.close,
                            prior.baseVolume,prior.quoteVolume,prior.trades,
                            prior.takerBase,prior.takerQuote)
                    }))
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
