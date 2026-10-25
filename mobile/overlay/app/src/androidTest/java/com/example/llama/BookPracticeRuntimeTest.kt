package com.example.llama

import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONArray
import java.io.File

@RunWith(AndroidJUnit4::class)
class BookPracticeRuntimeTest {
    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext
    private fun fixture(body: (File,File,StudyMemory.Passage,BookPractice.Case)->Unit) {
        val books=File.createTempFile("book-practice-test-",".db",ctx.cacheDir)
        val memory=File.createTempFile("practice-memory-test-",".db",ctx.cacheDir)
        try {
            val hashes=mapOf("TC1" to "36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c",
                "NTD" to "855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4",
                "TC2" to "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f",
                "COURSE" to "fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca")
            SQLiteDatabase.openOrCreateDatabase(books,null).use { db ->
                db.execSQL("CREATE TABLE volumes(book TEXT,sha256 TEXT,total_pages INTEGER,extracted_pages INTEGER,missing_pages INTEGER)")
                db.execSQL("CREATE TABLE excerpts(book TEXT,pdf_page INTEGER,content TEXT)")
                hashes.forEach { (ed,hash) ->
                    val pages=when(ed){"TC1"->140;"NTD"->156;"TC2"->224;else->122}
                    db.execSQL("INSERT INTO volumes VALUES(?,?,?,1,?)",arrayOf<Any>(ed,hash,pages,pages-1))
                    db.execSQL("INSERT INTO excerpts VALUES(?,10,?)",arrayOf(ed,"Фрактал подтверждается по закрытым свечам. Это синтетический источник теста."))
                }
            }
            val p=StudyMemory.next(books,memory,"TC2","model-a")!!
            val c=BookPractice.Case("fixture","TC2","TC2.WM3",listOf(10),"Когда подтверждается фрактал?",
                mapOf("A" to "По закрытым свечам","B" to "Заранее","C" to "Неизвестно"),"A")
            body(books,memory,p,c)
        } finally { books.delete();memory.delete() }
    }
    @Test fun packagedCurriculumContainsAllFourBooksWithoutLeakingKeys() {
        val cases=BookPractice.load(ctx)
        assertEquals(46,cases.size);assertEquals(setOf("TC1","NTD","TC2","COURSE"),cases.map{it.book}.toSet())
        val prompt=BookPractice.prompt(BookPractice.instructions(ctx),cases.first(),null)
        assertFalse(prompt.contains("\"correct\""));assertFalse(prompt.contains("feedback"))
        assertTrue(prompt.endsWith("/no_think"))
    }
    @Test fun actualAndroidPromptMatchesTheCpuDiagnosticForAllSixteenInputs() {
        val fixture=InstrumentationRegistry.getInstrumentation().context.assets
            .open("mobile_tutor_golden.json").bufferedReader().use { JSONArray(it.readText()) }
        val cases=BookPractice.load(ctx).associateBy { it.id }
        for (i in 0 until fixture.length()) {
            val row=fixture.getJSONObject(i)
            val source=if (row.isNull("source")) null else row.getJSONObject("source").let {
                val ed=it.getString("book");val page=it.getInt("page");val text=it.getString("text")
                StudyMemory.Passage(StudyMemory.key(ed,page,text),ed,page,text)
            }
            assertEquals(row.getString("prompt"),BookPractice.prompt(BookPractice.instructions(ctx),cases.getValue(row.getString("id")),source))
        }
    }
    @Test fun strictAnswerProtocolRejectsAmbiguousOrPartialOutput() {
        assertEquals("B",BookPractice.parse("{\"choice\":\"B\",\"reason\":\"source\"}").choice)
        for (raw in listOf("{\"choice\":\"A\",\"choice\":\"B\",\"reason\":\"x\"}",
            "<think>unfinished", "{\"choice\":\"A\",\"reason\":\"x\"} trailing",
            "{choice:A,reason:x}")) {
            try { BookPractice.parse(raw);fail("invalid protocol accepted") } catch (_: IllegalArgumentException) {}
        }
    }
    @Test fun mistakesSurviveReopenAndRecallOnlyCurrentSameModelSource() = fixture { books,memory,p,c ->
        BookPractice.record(memory,"model-a",p,c,BookPractice.Answer("B","wrong fixture"))
        assertEquals(0,BookPractice.score(memory,"model-a").correct)
        assertEquals(1,BookPractice.score(memory,"model-a").graded)
        assertTrue(BookPractice.recallSources(memory,books,"model-a","TC2","фрактал",listOf(c)).isNotEmpty())
        assertTrue(BookPractice.recallSources(memory,books,"model-a","TC2","Когда начинать?",listOf(c)).isEmpty())
        assertTrue(BookPractice.recallSources(memory,books,"model-b","TC2","фрактал",listOf(c)).isEmpty())
        assertTrue(BookPractice.recallSources(memory,books,"model-a","NTD","фрактал",listOf(c)).isEmpty())
        val revised=BookPractice.Case(c.id,c.book,c.rule,c.pages,c.question,c.choices,"B")
        assertTrue(BookPractice.recallSources(memory,books,"model-a","TC2","фрактал",listOf(revised)).isEmpty())
        SQLiteDatabase.openDatabase(books.absolutePath,null,SQLiteDatabase.OPEN_READWRITE).use { db ->
            db.execSQL("UPDATE excerpts SET content='Исправленный OCR фрактала' WHERE book='TC2'")
        }
        assertTrue(BookPractice.recallSources(memory,books,"model-a","TC2","фрактал",listOf(c)).isEmpty())
    }
    @Test fun interfaceFailureIsSeparateFromWrongAnswerAndBadReadingDoesNotBlockOtherBooks() = fixture { books,memory,p,c ->
        BookPractice.record(memory,"model-a",p,c,null)
        assertEquals(0,BookPractice.score(memory,"model-a").graded)
        assertEquals(1,BookPractice.score(memory,"model-a").protocolErrors)
        repeat(2) { StudyMemory.failed(memory,p,"model-a") }
        assertEquals(null,StudyMemory.next(books,memory,"TC2","model-a"))
        assertNotNull(StudyMemory.next(books,memory,"NTD","model-a"))
        StudyMemory.beginSession(memory,"model-a")
        assertNotNull(StudyMemory.next(books,memory,"TC2","model-a"))
    }
    @Test fun savingReadingBeforeAnInterruptionDoesNotLosePendingPractice() = fixture { books,memory,p,c ->
        StudyMemory.save(memory,p,"model-a","{\"summary\":\"Фрактал\",\"support\":\"Фрактал подтверждается по закрытым свечам.\"}")
        assertEquals(null,StudyMemory.next(books,memory,"TC2","model-a"))
        assertNotNull(BookPractice.pendingSource(books,memory,"model-a","TC2",listOf(c)))
        BookPractice.record(memory,"model-a",p,c,BookPractice.Answer("A","source"))
        assertEquals(null,BookPractice.pendingSource(books,memory,"model-a","TC2",listOf(c)))
    }
}
