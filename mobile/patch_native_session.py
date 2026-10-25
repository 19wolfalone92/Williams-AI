"""Reviewed patches for ONE pinned llama.android; no model or network access."""
from hashlib import sha1
from pathlib import Path
import sys


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("PINNED_ANDROID_PATCH_ANCHOR_CHANGED")
    return text.replace(old, new)


def patch(project: Path):
    files = {
        "lib/src/main/java/com/arm/aichat/InferenceEngine.kt": "26c1668724662f9fb57f93f725e36d5678a4aced",
        "lib/src/main/java/com/arm/aichat/internal/InferenceEngineImpl.kt": "a293796f9017326531021c9be38bdd33fcb9667e",
        "lib/src/main/cpp/ai_chat.cpp": "ca6f7172d910da06a26be0eaaa3da1ff01834639",
    }
    originals = {}
    for name, expected in files.items():
        raw = (project / name).read_bytes()
        if sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != expected:
            raise ValueError("PINNED_ANDROID_SOURCE_BLOB_CHANGED:" + name)
        originals[name] = raw.decode()
    interface, implementation, native = files
    changed = {}
    changed[interface] = replace(originals[interface],
        "    suspend fun loadModel(pathToModel: String)",
        "    suspend fun loadModel(pathToModel: String)\n\n"
        "    /** Clear conversation/KV/sampling state while retaining loaded weights. */\n"
        "    suspend fun resetSession()")
    text = replace(originals[implementation], "    private external fun generateNextToken(): String?",
        "    private external fun generateNextToken(): String?\n\n"
        "    @FastNative\n    private external fun resetConversation(): Int\n\n"
        "    @FastNative\n    private external fun wasGenerationTruncated(): Boolean")
    text = replace(text, "    /**\n     * Process the plain text system prompt",
        "    override suspend fun resetSession() = withContext(llamaDispatcher) {\n"
        "        check(_state.value is InferenceEngine.State.ModelReady ||\n"
        "              _state.value is InferenceEngine.State.Error) { \"MODEL_NOT_IDLE\" }\n"
        "        check(resetConversation() == 0) { \"NO_LOADED_MODEL_FOR_SESSION\" }\n"
        "        _cancelGeneration = false\n        _readyForSystemPrompt = true\n"
        "        _state.value = InferenceEngine.State.ModelReady\n    }\n\n"
        "    /**\n     * Process the plain text system prompt")
    text = replace(text, '                    Log.e(TAG, "Failed to process user prompt: $result")\n                    return@flow',
        '                    throw IOException(if (result == 3) "MODEL_CONTEXT_BUDGET_EXCEEDED" else "MODEL_PROMPT_DECODE_FAILED")')
    text = replace(text, "            if (_cancelGeneration) {",
        '            if (!_cancelGeneration && wasGenerationTruncated()) throw IOException("MODEL_OUTPUT_TRUNCATED")\n'
        "            if (_cancelGeneration) {")
    changed[implementation] = text
    text = replace(originals[native], "static llama_pos stop_generation_position;",
        "static llama_pos stop_generation_position;\nstatic bool generation_truncated = false;")
    text = replace(text, "    stop_generation_position = 0;",
        "    stop_generation_position = 0;\n    generation_truncated = false;")
    anchor = 'extern "C"\nJNIEXPORT jint JNICALL\nJava_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt('
    text = replace(text, anchor,
        'extern "C"\nJNIEXPORT jint JNICALL\n'
        'Java_com_arm_aichat_internal_InferenceEngineImpl_resetConversation(JNIEnv *, jobject) {\n'
        '    if (!g_model || !g_context || !g_sampler) return 1;\n'
        '    reset_long_term_states();\n    reset_short_term_states();\n'
        '    common_sampler_reset(g_sampler);\n    return 0;\n}\n\n'
        'extern "C"\nJNIEXPORT jboolean JNICALL\n'
        'Java_com_arm_aichat_internal_InferenceEngineImpl_wasGenerationTruncated(JNIEnv *, jobject) {\n'
        '    return generation_truncated ? JNI_TRUE : JNI_FALSE;\n}\n\n' + anchor)
    text = replace(text, "    if (user_prompt_size > max_batch_size) {\n"
        "        const int skipped_tokens = user_prompt_size - max_batch_size;\n"
        "        user_tokens.resize(max_batch_size);\n"
        '        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);\n    }',
        "    if (n_predict <= 0 || current_position + user_prompt_size + n_predict > max_batch_size) {\n"
        '        LOGw("Independent prompt and output exceed context budget");\n        return 3;\n    }')
    text = replace(text, "stop_generation_position = current_position + user_prompt_size + n_predict;",
        "stop_generation_position = current_position + n_predict;")
    text = replace(text, "    if (current_position >= stop_generation_position) {",
        "    if (current_position >= stop_generation_position) {\n        generation_truncated = true;")
    changed[native] = text
    for name, text in changed.items():
        (project / name).write_text(text)


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
