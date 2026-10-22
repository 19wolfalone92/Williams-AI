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

    fun retrieve(file: File, question: String, limit: Int = 3): List<BookExcerpt> {
        if (!file.isFile) return emptyList()
        require(limit in 1..5)
        val tokens = Regex("[\\p{L}\\p{N}]{3,}")
            .findAll(question.lowercase()).map { it.value }.distinct().take(9).toList()
        if (tokens.isEmpty()) return emptyList()
        val query = tokens.joinToString(" OR ") { "\"" + it + "\"" }
        SQLiteDatabase.openDatabase(file.absolutePath, null, SQLiteDatabase.OPEN_READONLY).use { db ->
            db.rawQuery(
                "SELECT book,pdf_page,substr(content,1,650) " +
                "FROM excerpts WHERE excerpts MATCH ? ORDER BY rank LIMIT " + limit,
                arrayOf(query)
            ).use { cursor ->
                val found = ArrayList<BookExcerpt>()
                while (cursor.moveToNext()) {
                    val id = cursor.getString(0)
                    val page = cursor.getInt(1)
                    if (id !in expected || page <= 0) continue
                    found.add(BookExcerpt(
                        id, page, cursor.getString(2).replace(Regex("\\s+"), " ").take(650)
                    ))
                }
                return found
            }
        }
    }

    fun cite(excerpts: List<BookExcerpt>): String = excerpts.joinToString("\n\n") {
        "[" + it.book + ", PDF с. " + it.page + "] Извлечённый текст (непроверенная машинная копия): " + it.excerpt
    }
}
