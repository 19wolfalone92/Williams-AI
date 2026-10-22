package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID

/** Independent offline chat: no Binance account, no network API, no order execution. */
class MainActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var list: RecyclerView
    private lateinit var input: EditText
    private lateinit var button: FloatingActionButton
    private var inference: InferenceEngine? = null
    private var generation: Job? = null
    private var modelReady = false
    private var busy = true
    private val messages = mutableListOf<Message>()
    private val adapter = MessageAdapter(messages)
    private val response = StringBuilder()
    private val modelPicker = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) lifecycleScope.launch { importModel(uri) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        status = findViewById(R.id.gguf)
        list = findViewById(R.id.messages)
        input = findViewById(R.id.user_input)
        button = findViewById(R.id.fab)
        list.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        list.adapter = adapter
        input.isEnabled = false
        button.isEnabled = false
        setStatus("Williams AI: запуск локального движка…")
        button.setOnClickListener {
            if (!busy) {
                if (modelReady) ask() else modelPicker.launch(arrayOf("*/*"))
            }
        }
        lifecycleScope.launch {
            try {
                inference = withContext(Dispatchers.Default) {
                    AiChat.getInferenceEngine(applicationContext)
                }
                val saved = getPreferences(MODE_PRIVATE).getString("gguf_name", null)
                val path = saved?.takeIf { HASH_NAME.matches(it) }?.let {
                    File(File(filesDir, "models"), it)
                }
                if (path?.isFile == true) loadModel(path)
                else {
                    busy = false
                    button.isEnabled = true
                    setStatus("Williams AI — без облака и API.\nНажмите кнопку для выбора локальной GGUF-модели.")
                }
            } catch (e: Exception) {
                busy = false
                button.isEnabled = true
                setStatus("Ошибка движка: " + e.message)
            }
        }
    }

    private suspend fun importModel(uri: Uri) {
        if (busy) return
        busy = true
        button.isEnabled = false
        input.isEnabled = false
        setStatus("Копирование GGUF в закрытое хранилище…")
        try {
            val file = withContext(Dispatchers.IO) { verifiedCopy(uri) }
            getPreferences(MODE_PRIVATE).edit().putString("gguf_name", file.name).apply()
            loadModel(file)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            busy = false
            button.isEnabled = true
            setStatus("Ошибка импорта: " + (e.message ?: "неизвестная"))
        }
    }

    /** SHA-256 name + temporary file prevent filename injection and partial models. */
    private fun verifiedCopy(uri: Uri): File {
        val models = File(filesDir, "models")
        require(models.isDirectory || models.mkdirs()) { "Нет каталога моделей" }
        val partial = File.createTempFile("import-", ".partial", models)
        try {
            val digest = MessageDigest.getInstance("SHA-256")
            val source = contentResolver.openInputStream(uri) ?: error("Файл недоступен")
            source.use { stream ->
                FileOutputStream(partial).use { target ->
                    val buf = ByteArray(65536)
                    var total = 0L
                    var prefix = 0
                    val magic = byteArrayOf(0x47, 0x47, 0x55, 0x46)
                    while (true) {
                        val n = stream.read(buf)
                        if (n < 0) break
                        for (i in 0 until n) {
                            if (prefix < 4) {
                                require(buf[i] == magic[prefix]) { "Это не GGUF-модель" }
                                prefix++
                            } else break
                        }
                        total += n
                        require(total <= 6L * 1024L * 1024L * 1024L) { "Модель более 6 ГБ" }
                        digest.update(buf, 0, n)
                        target.write(buf, 0, n)
                    }
                    require(prefix == 4 && total >= 1024L) { "GGUF-файл повреждён или пуст" }
                    target.fd.sync()
                }
            }
            val sha = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
            val destination = File(models, "$sha.gguf")
            if (destination.exists()) {
                require(destination.length() == partial.length()) { "Конфликт имени модели" }
                return destination
            }
            check(partial.renameTo(destination)) { "Не удалось сохранить GGUF" }
            return final
        } finally {
            partial.delete()
        }
    }

    private suspend fun loadModel(file: File) {
        busy = true
        button.isEnabled = false
        setStatus("Загрузка модели в память…")
        try {
            val engine = inference ?: error("Локальный движок ещё не запущен")
            withContext(Dispatchers.Default) { engine.loadModel(file.absolutePath) }
            modelReady = true
            setStatus("Williams AI — готов. Режим OFFLINE.\nМодель: " +
                file.name.take(12) + "…\nБез доступа к Binance. Ответы не являются торговыми приказами.")
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            modelReady = false
            getPreferences(MODE_PRIVATE).edit().remove("gguf_name").apply()
            setStatus("Не удалось загрузить модель: " + (e.message ?: "ошибка"))
        } finally {
            busy = false
            button.isEnabled = true
            input.isEnabled = modelReady
            if (modelReady) {
                input.hint = "Спросите Williams AI…"
                button.setImageResource(R.drawable.outline_send_24)
            } else {
                input.hint = "Выберите GGUF"
                button.setImageResource(R.drawable.outline_folder_open_24)
            }
        }
    }

    private fun ask() {
        val question = input.text.toString().trim()
        if (question.isEmpty()) return
        if (question.length > 1200) {
            Toast.makeText(this, "Максимум 1200 символов", Toast.LENGTH_LONG).show()
            return
        }
        val engine = inference ?: return
        input.text.clear()
        busy = true
        input.isEnabled = false
        button.isEnabled = false
        messages.add(Message(UUID.randomUUID().toString(), question, true))
        response.clear()
        messages.add(Message(UUID.randomUUID().toString(), "", false))
        adapter.notifyItemRangeInserted(messages.size - 2, 2)
        list.scrollToPosition(messages.lastIndex)
        val prompt = "Ты Williams AI — русскоязычный локальный аналитик. " +
            "Не выдумывай факты, котировки, свечи и цитаты из книг. " +
            "Без фактических рыночных данных не утверждай, что видел график. " +
            "Различай факты и гипотезы. Не отдавай приказов на реальные сделки. " +
            "Твои ответы предназначены только для исследования.\nВопрос: " + question
        generation = lifecycleScope.launch {
            try {
                withContext(Dispatchers.Default) {
                    engine.sendUserPrompt(prompt).collect { token ->
                        withContext(Dispatchers.Main) {
                            response.append(token)
                            val idx = messages.lastIndex
                            messages[idx] = messages[idx].copy(content = response.toString())
                            adapter.notifyItemChanged(idx)
                            list.scrollToPosition(idx)
                        }
                    }
                }
            } catch (_: CancellationException) {
                // Cancelled when activity goes to background.
            } catch (e: Exception) {
                setStatus("Ошибка ответа ИИ: " + (e.message ?: "неизвестная"))
            } finally {
                busy = false
                input.isEnabled = modelReady
                button.isEnabled = true
            }
        }
    }

    private fun setStatus(message: String) { status.text = message }
    override fun onStop() {
        generation?.cancel()
        super.onStop()
    }
    override fun onDestroy() {
        generation?.cancel()
        inference?.destroy()
        inference = null
        super.onDestroy()
    }
    companion object {
        private val HASH_NAME = Regex("^[0-9a-f]{64}\\.gguf$")
    }
}
