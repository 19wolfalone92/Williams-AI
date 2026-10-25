package com.example.llama

import com.arm.aichat.InferenceEngine
import kotlinx.coroutines.flow.collect

/** One loaded Qwen, bounded independent tasks; durable memory lives in SQLite. */
object QwenSession {
    suspend fun answer(engine: InferenceEngine, prompt: String, tokens: Int, maxChars: Int,
                       onToken: suspend (String) -> Unit = {}): String {
        require(prompt.isNotBlank() && tokens in 32..2048 && maxChars in 128..16000)
        engine.resetSession()
        val output = StringBuilder()
        engine.sendUserPrompt(prompt, predictLength = tokens).collect { token ->
            check(output.length + token.length <= maxChars) { "MODEL_OUTPUT_CHAR_LIMIT" }
            output.append(token)
            onToken(token)
        }
        check(output.isNotBlank()) { "MODEL_EMPTY_RESPONSE" }
        return output.toString()
    }
}
