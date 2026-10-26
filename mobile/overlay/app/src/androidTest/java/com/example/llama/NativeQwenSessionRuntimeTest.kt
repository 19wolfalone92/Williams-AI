package com.example.llama

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.IOException
import java.security.MessageDigest

/** Real native inference with a 19 MB plumbing fixture, NOT Williams/Qwen skill. */
@RunWith(AndroidJUnit4::class)
class NativeQwenSessionRuntimeTest {
    @Test fun actualNativeResetBudgetAndErrorRecoveryWithoutReloadingWeights() { runBlocking {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val ctx=instrumentation.targetContext
        val model=File(ctx.cacheDir,"native-session-test.gguf")
        instrumentation.context.assets.open("native-runtime-test.gguf").use { source ->
            model.outputStream().use { source.copyTo(it) }
        }
        val digest=MessageDigest.getInstance("SHA-256").digest(model.readBytes())
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        assertEquals("66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",digest)
        val engine=AiChat.getInferenceEngine(ctx)
        try {
            withTimeout(30000) { engine.state.first { it is InferenceEngine.State.Initialized } }
            engine.loadModel(model.absolutePath)
            var truncated=0
            suspend fun one(prompt: String) {
                engine.resetSession()
                try { withTimeout(120000) { engine.sendUserPrompt(prompt,1).collect {} } }
                catch (e: IOException) {
                    assertEquals("MODEL_OUTPUT_TRUNCATED",e.message);truncated++
                }
            }
            // Each prompt occupies over half the native context. The second
            // can fit only if reset actually clears native positions and KV.
            one(" a".repeat(4100))
            one(" a".repeat(4100))
            one("Once upon a time there was a little")
            assertTrue("At least one real generation must hit its one-token limit",truncated>0)
            engine.resetSession()
            try {
                engine.sendUserPrompt(" a".repeat(10000),1).collect {}
                fail("oversized prompt must not be silently truncated")
            } catch (e: IOException) { assertEquals("MODEL_CONTEXT_BUDGET_EXCEEDED",e.message) }
            engine.resetSession()
            assertTrue(engine.state.value is InferenceEngine.State.ModelReady)
            one("Once upon a time")
            // Exactly one loadModel above; reset does not unload/reload weights.
        } finally { engine.destroy();model.delete() }
        // Activity recreation must get a fresh initialized singleton rather
        // than the destroyed engine with stale state and cancelled scope.
        val reopened=AiChat.getInferenceEngine(ctx)
        try {
            assertNotSame(engine,reopened)
            withTimeout(30000) { reopened.state.first { it is InferenceEngine.State.Initialized } }
        } finally { reopened.destroy() }
    }
    }
}
