package com.example.llama

import com.arm.aichat.InferenceEngine
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class QwenSessionTest {
    private class Engine : InferenceEngine {
        override val state = MutableStateFlow<InferenceEngine.State>(InferenceEngine.State.ModelReady)
        var resets = 0; var loads = 0; val seen = arrayListOf<Pair<String,Int>>()
        var output = "response"; var fail = false
        override suspend fun loadModel(pathToModel: String) { loads++ }
        override suspend fun resetSession() { resets++ }
        override suspend fun setSystemPrompt(systemPrompt: String) {}
        override fun sendUserPrompt(message: String, predictLength: Int): Flow<String> = flow {
            seen.add(message to predictLength)
            emit(output)
            if (fail) throw IllegalStateException("MODEL_OUTPUT_TRUNCATED")
        }
        override suspend fun bench(pp: Int,tg: Int,pl: Int,nr: Int) = ""
        override fun cleanUp() {}
        override fun destroy() {}
    }

    @Test fun independentTasksResetContextButDoNotReloadModel() = runBlocking {
        val e=Engine()
        QwenSession.answer(e,"first source",384,4000)
        QwenSession.answer(e,"second source",1024,6000)
        assertEquals(2,e.resets);assertEquals(0,e.loads)
        assertEquals(listOf("first source" to 384,"second source" to 1024),e.seen)
    }
    @Test fun interruptedCompleteLookingJsonIsNotReturnedAsALesson() = runBlocking {
        val e=Engine();e.output="{\"choice\":\"A\",\"reason\":\"complete looking\"}";e.fail=true
        try { QwenSession.answer(e,"source",384,4000);fail("must reject truncation") }
        catch (failure: IllegalStateException) { assertEquals("MODEL_OUTPUT_TRUNCATED",failure.message) }
    }
    @Test fun emptyAndOversizedResponsesFailExplicitly() = runBlocking {
        val e=Engine()
        for (text in listOf("","x".repeat(129))) {
            e.output=text
            try { QwenSession.answer(e,"source",384,128);fail("must reject invalid response") }
            catch (_: IllegalStateException) {}
        }
    }
}
