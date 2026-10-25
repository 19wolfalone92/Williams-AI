package com.example.llama

import android.database.sqlite.SQLiteDatabase
import java.io.File

/** Private, offline retrieval of user's licensed Williams page index. */
object BookKnowledge {
    private val expected = mapOf(
        "TC1" to "36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c",
        "NTD" to "855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4",
        "TC2" to "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f",
        "COURSE" to "fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca"
    )
    data class BookExcerpt(val book: String, val page: Int, val excerpt: String)

    fun validate(file: File): Map<String, Int> {
        require(file.isFile && file.length() in 1000..(50L * 1024 * 1024)) {
            "Требуется приватный SQLite индекс книг Williams AI"
        }
        val validated = linkedMapOf<String, Int>()
        SQLiteDatabase.openDatabase(
            file.absolutePath, null, SQLiteDatabase.OPEN_READONLY
        ).use { db ->
            db.rawQuery("SELECT sql FROM sqlite_master WHERE name='excerpts'", null).use { schema ->
                require(schema.moveToFirst() && !schema.getString(0).contains("VIRTUAL TABLE", ignoreCase = true)) {
                    "Старый FTS-индекс: загрузите новый переносимый SQLite-файл, совместимый с Android"
                }
            }
            db.rawQuery(
                "SELECT book,sha256,total_pages,extracted_pages,missing_pages FROM volumes", null
            ).use { cur ->
                while (cur.moveToNext()) {
                    val book = cur.getString(0)
                    val checksum = cur.getString(1)
                    val total = cur.getInt(2)
                    val extracted = cur.getInt(3)
                    val missing = cur.getInt(4)
                    val count = when (book) {
                        "TC1" -> 140
                        "NTD" -> 156
                        "TC2" -> 224
                        "COURSE" -> 122
                        else -> 0
                    }
                    require(count > 0 && !validated.containsKey(book) &&
                        checksum == expected[book] && total == count &&
                        extracted in 0..count && missing == count - extracted) {
                        "Несовпадение происхождения первоисточников"
                    }
                    validated[book] = extracted
                }
            }
            require(validated.keys == expected.keys) {
                "Индекс должен содержать паспорт всех четырёх книг"
            }
            db.rawQuery("SELECT count(*) FROM excerpts", null).use { cur ->
                require(cur.moveToFirst() && cur.getInt(0) > 0) {
                    "База не содержит доступных страниц"
                }
            }
        }
        return validated
    }

    fun retrieve(file: File, question: String, limit: Int = 3, edition: String? = null): List<BookExcerpt> {
        if (!file.isFile) return emptyList()
        require(limit in 1..5)
        require(edition == null || edition in expected)
        validate(file)
        val original = Regex("[\\p{L}\\p{N}]{3,}")
            .findAll(question.lowercase(java.util.Locale.ROOT)).map { it.value }.distinct().take(9).toList()
        val groups = listOf(listOf("фракт", "fractal"), listOf("ангуля", "angulation"),
            listOf("аллигатор", "alligator"), listOf("мудрец", "wise"),
            listOf("стоп", "stop"), listOf("волн", "wave"), listOf("диверген", "divergenc"),
            listOf("баланс", "balance"), listOf("пирамид", "pyramid"), listOf("риск", "risk"))
        val aliases = groups.filter { group -> original.any { word -> group.any { word.startsWith(it) } } }.flatten()
        val tokens = (aliases + original).distinct().take(20)
        if (tokens.isEmpty()) return emptyList()
        SQLiteDatabase.openDatabase(file.absolutePath, null, SQLiteDatabase.OPEN_READONLY).use { db ->
            db.rawQuery(
                "SELECT book,pdf_page,content FROM excerpts" +
                    (if (edition == null) "" else " WHERE book=?") +
                    " ORDER BY book,CAST(pdf_page AS INTEGER)",
                if (edition == null) null else arrayOf(edition)
            ).use { cursor ->
                val found = ArrayList<Pair<Int, BookExcerpt>>()
                while (cursor.moveToNext()) {
                    val id = cursor.getString(0)
                    val page = cursor.getInt(1)
                    if (id !in expected || page <= 0) continue
                    val content = cursor.getString(2).replace(Regex("\\s+"), " ")
                    val lower = content.lowercase(java.util.Locale.ROOT)
                    val words = Regex("[\\p{L}\\p{N}]{2,}").findAll(lower).map { it.value }.toList()
                    val score = tokens.sumOf { token -> minOf(4, words.count { it.startsWith(token) }) }
                    if (score == 0) continue
                    val offset = tokens.map { lower.indexOf(it) }.filter { it >= 0 }.minOrNull() ?: 0
                    found.add(Pair(score, BookExcerpt(id, page, content.drop(maxOf(0, offset - 120)).take(650))))
                    if (found.size > 20) {
                        found.sortByDescending { it.first }
                        while (found.size > limit) found.removeAt(found.lastIndex)
                    }
                }
                return found.sortedByDescending { it.first }.take(limit).map { it.second }
            }
        }
    }

    fun cite(excerpts: List<BookExcerpt>): String = excerpts.joinToString("\n\n") {
        "[" + it.book + ", PDF с. " + it.page + "] Извлечённый текст (непроверенная машинная копия): " + it.excerpt
    }

    /** Only a final answer can request study; hidden thinking is not a decision. */
    fun finalAnswer(raw: String): String {
        var text = raw.trim()
        if (text.startsWith("<think>")) {
            val end = text.indexOf("</think>")
            if (end < 0 || text.indexOf("<think>", 7) >= 0) return ""
            text = text.substring(end + 8).trim()
        }
        if (text.contains("<think>") || text.contains("</think>")) return ""
        return text
    }

    fun reviewTopic(answer: String): String? {
        val final = finalAnswer(answer)
        val explicit = Regex("\\[НУЖНА КНИГА: (ENTRY|MANAGEMENT|EXIT|WAVE)\\]", RegexOption.IGNORE_CASE)
            .find(final)?.groupValues?.get(1)
        return explicit?.uppercase(java.util.Locale.ROOT)
    }

    fun reviewQuery(question: String, topic: String): String {
        val words = when (topic) {
            "ENTRY" -> "вход пробой подтверждение фрактал entry trigger"
            "MANAGEMENT" -> "сопровождение перенос стоп stop trailing pyramid"
            "EXIT" -> "выход закрытие стоп противоположный exit stop"
            "WAVE" -> "волна степень вложенность коррекция wave Elliott"
            else -> error("Неизвестная тема перечитывания")
        }
        return words + " " + question.take(450)
    }

    fun citationsSupported(answer: String, excerpts: List<BookExcerpt>): Boolean {
        val allowed = excerpts.map { it.book + ":" + it.page }.toSet()
        val citations = Regex("(TC1|NTD|TC2|COURSE)[,:]?\\s*PDF\\s*(?:с\\.?|стр\\.?|page)?\\s*(\\d+)",
            RegexOption.IGNORE_CASE).findAll(finalAnswer(answer))
        return citations.all {
            it.groupValues[1].uppercase(java.util.Locale.ROOT) + ":" + it.groupValues[2] in allowed
        }
    }
}
