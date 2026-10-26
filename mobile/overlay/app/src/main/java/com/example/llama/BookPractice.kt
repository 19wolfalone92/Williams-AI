package com.example.llama

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import org.json.JSONObject
import org.json.JSONTokener
import java.io.File
import java.security.MessageDigest
import java.util.Locale

/** Source-assisted exercises authored by the project; never expert trading labels. */
object BookPractice {
    data class Case(val id: String, val book: String, val rule: String, val pages: List<Int>,
                    val question: String, val choices: Map<String, String>, val correct: String) {
        val key: String get() = id + ":" + MessageDigest.getInstance("SHA-256").digest(
            (listOf(book,rule,pages.toString(),question) + listOf("A","B","C").map { choices.getValue(it) } + correct)
                .joinToString("\u0000").toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
    data class Answer(val choice: String, val reason: String)
    data class Score(val correct: Int, val graded: Int, val protocolErrors: Int)

    fun load(ctx: Context): List<Case> {
        fun asset(name: String) = ctx.assets.open(name).bufferedReader().use { it.readText() }
        val rules = JSONObject(asset("SOURCE_RULE_CARDS.json")).getJSONArray("rule_cards")
        val cards = (0 until rules.length()).associate { rules.getJSONObject(it).let { r -> r.getString("id") to r } }
        val rows = JSONObject(asset("ADAPTIVE_TUTOR_PRACTICE_RU.json")).getJSONArray("cases")
        val seen = HashSet<String>()
        return (0 until rows.length()).map { i ->
            val row = rows.getJSONObject(i)
            val rule = cards.getValue(row.getString("rule_id"))
            val book = row.getString("edition")
            require(book in setOf("TC1", "NTD", "TC2", "COURSE") &&
                rule.getString("source") == book && rule.getString("status") == "REVIEWED_PAGES")
            val pages = rule.getJSONArray("pdf_pages")
            val id = row.getString("id")
            require(seen.add(id))
            val choices = row.getJSONObject("choices")
            require(choices.length() == 3)
            val options = listOf("A", "B", "C").associateWith { choices.getString(it) }
            val correct = row.getString("correct")
            require(correct in options)
            Case(id, book, row.getString("rule_id"), (0 until pages.length()).map { pages.getInt(it) },
                row.getString("question"), options, correct)
        }
    }

    fun instructions(ctx: Context): String = ctx.assets.open("MOBILE_TUTOR_PROMPT_RU.txt")
        .bufferedReader().use { it.readText().trim() }

    fun prompt(instructions: String, case: Case, source: StudyMemory.Passage?): String {
        require(instructions.length in 1..1200 && case.question.length in 1..1200)
        if (source != null) require(source.book == case.book && source.page in case.pages &&
            source.text.length in 12..3000 && source.key == StudyMemory.key(source.book, source.page, source.text))
        // Grading key/feedback are never rendered. Raw source is explicitly data.
        return instructions + "\nИздание: " + case.book + "\nВопрос: " + case.question +
            listOf("A", "B", "C").joinToString("") { "\n" + it + ": " + case.choices.getValue(it) } +
            "\nИсточник:\n" + (source?.let { "[" + it.book + ", PDF с. " + it.page + "]\n" + it.text }
                ?: "Не предоставлен; ответь по имеющимся знаниям.") + "\n/no_think"
    }

    fun parse(raw: String): Answer {
        require(raw.length in 1..4000)
        var text = BookKnowledge.finalAnswer(raw)
        if (text.startsWith("```json\n") && text.endsWith("\n```"))
            text = text.removePrefix("```json\n").removeSuffix("\n```").trim()
        val tokens = JSONTokener(text)
        require(tokens.nextClean() == '{')
        val fields = linkedMapOf<String, String>()
        while (true) {
            require(tokens.nextClean() == '"'); tokens.back()
            val key = tokens.nextValue() as String
            require(key !in fields && key in setOf("choice", "reason") && tokens.nextClean() == ':')
            require(tokens.nextClean() == '"'); tokens.back()
            fields[key] = tokens.nextValue() as String
            val end = tokens.nextClean()
            if (end == '}') break
            require(end == ',')
        }
        require(tokens.nextClean() == '\u0000' && fields.keys == setOf("choice", "reason"))
        require(fields.getValue("choice") in setOf("A", "B", "C") && fields.getValue("reason").length in 1..180)
        return Answer(fields.getValue("choice"), fields.getValue("reason"))
    }

    private fun open(memory: File): SQLiteDatabase {
        val db = SQLiteDatabase.openOrCreateDatabase(memory, null)
        db.execSQL("CREATE TABLE IF NOT EXISTS book_practice (seq INTEGER PRIMARY KEY AUTOINCREMENT," +
            "source_key TEXT NOT NULL,model_id TEXT NOT NULL,book TEXT NOT NULL,page INTEGER NOT NULL," +
            "case_id TEXT NOT NULL,rule_id TEXT NOT NULL,choice TEXT,correct INTEGER NOT NULL," +
            "attempted_at INTEGER NOT NULL,UNIQUE(source_key,model_id,case_id))")
        return db
    }

    fun next(memory: File, modelId: String, p: StudyMemory.Passage, cases: List<Case>): Case? =
        open(memory).use { db -> cases.firstOrNull { c -> c.book == p.book && p.page in c.pages &&
            !db.rawQuery("SELECT 1 FROM book_practice WHERE source_key=? AND model_id=? AND case_id=?",
                arrayOf(p.key, modelId, c.key)).use { it.moveToFirst() } } }

    fun record(memory: File, modelId: String, p: StudyMemory.Passage, case: Case, answer: Answer?) {
        require(modelId.isNotBlank() && modelId.length <= 160 && case.book == p.book && p.page in case.pages &&
            p.key == StudyMemory.key(p.book, p.page, p.text))
        val correct = if (answer == null) -1 else if (answer.choice == case.correct) 1 else 0
        open(memory).use { db -> db.execSQL("INSERT OR IGNORE INTO book_practice " +
            "(source_key,model_id,book,page,case_id,rule_id,choice,correct,attempted_at) VALUES(?,?,?,?,?,?,?,?,?)",
            arrayOf<Any?>(p.key,modelId,p.book,p.page,case.key,case.rule,answer?.choice,correct,System.currentTimeMillis())) }
    }

    /** Resume practice if the process stopped after saving a reading card. */
    fun pendingSource(books: File, memory: File, modelId: String, edition: String,
                      cases: List<Case>): StudyMemory.Passage? {
        BookKnowledge.validate(books)
        val saved = open(memory).use { db ->
            db.rawQuery("SELECT n.source_key FROM notes n WHERE n.model_id=? AND n.book=? AND NOT EXISTS(" +
                "SELECT 1 FROM reading_failures f WHERE f.model_id=n.model_id AND f.source_key=n.source_key AND f.failures>=2)",
                arrayOf(modelId,edition)).use { cur ->
                val keys=HashSet<String>();while(cur.moveToNext()) keys.add(cur.getString(0));keys
            }
        }
        if(saved.isEmpty()) return null
        SQLiteDatabase.openDatabase(books.absolutePath,null,SQLiteDatabase.OPEN_READONLY).use { db ->
            db.rawQuery("SELECT pdf_page,content FROM excerpts WHERE book=? ORDER BY rowid",arrayOf(edition)).use { cur ->
                while(cur.moveToNext()) {
                    val page=cur.getInt(0);val text=cur.getString(1);val key=StudyMemory.key(edition,page,text)
                    if(key !in saved) continue
                    val p=StudyMemory.Passage(key,edition,page,text)
                    if(next(memory,modelId,p,cases)!=null) return p
                }
            }
        }
        return null
    }

    fun score(memory: File, modelId: String): Score = open(memory).use { db ->
        db.rawQuery("SELECT sum(correct=1),sum(correct>=0),sum(correct<0) FROM book_practice WHERE model_id=?",
            arrayOf(modelId)).use { cur -> cur.moveToFirst(); Score(cur.getInt(0),cur.getInt(1),cur.getInt(2)) }
    }

    /** Re-read current original fragments for unresolved mistakes, never serve answer keys. */
    fun recallSources(memory: File, books: File, modelId: String, edition: String,
                      question: String, cases: List<Case>): List<BookKnowledge.BookExcerpt> {
        if (!memory.isFile || !books.isFile) return emptyList()
        BookKnowledge.validate(books)
        val generic = setOf("как","что","это","когда","почему","чего","какой","какая","какие","какую",
            "можно","нужно","для","при","или","про","пожалуйста","расскажи","покажи","объясни",
            "the","and","what","when","how","tc1","tc2","ntd","course")
        val words = Regex("[\\p{L}\\p{N}]{3,}").findAll(question.lowercase(Locale.ROOT))
            .map { it.value }.filter { it !in generic }.distinct().take(12).toList()
        if (words.isEmpty()) return emptyList()
        val byId = cases.filter { it.book == edition }.associateBy { it.key }
        val mistakes = open(memory).use { db ->
            db.rawQuery("SELECT p.source_key,p.case_id FROM book_practice p WHERE p.model_id=? AND p.book=? " +
                "AND p.correct=0 AND NOT EXISTS(SELECT 1 FROM book_practice n WHERE n.model_id=p.model_id " +
                "AND n.book=p.book AND n.case_id=p.case_id AND n.correct>=0 AND n.seq>p.seq) " +
                "ORDER BY p.seq DESC LIMIT 200", arrayOf(modelId,edition)).use { cur ->
                val result = linkedSetOf<String>()
                while (cur.moveToNext()) {
                    val c = byId[cur.getString(1)] ?: continue
                    if (words.any { c.question.lowercase(Locale.ROOT).contains(it) || c.rule.lowercase(Locale.ROOT).contains(it) })
                        result.add(cur.getString(0))
                }
                result
            }
        }
        if (mistakes.isEmpty()) return emptyList()
        val result = ArrayList<BookKnowledge.BookExcerpt>()
        SQLiteDatabase.openDatabase(books.absolutePath,null,SQLiteDatabase.OPEN_READONLY).use { db ->
            db.rawQuery("SELECT pdf_page,content FROM excerpts WHERE book=? ORDER BY rowid",arrayOf(edition)).use { cur ->
                while (cur.moveToNext()) {
                    val page=cur.getInt(0); val text=cur.getString(1)
                    if (StudyMemory.key(edition,page,text) in mistakes)
                        result.add(BookKnowledge.BookExcerpt(edition,page,text.replace(Regex("\\s+")," ").take(650)))
                    if (result.size == 2) break
                }
            }
        }
        return result
    }
}
