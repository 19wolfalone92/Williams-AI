package com.example.llama

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context
import java.util.concurrent.atomic.AtomicBoolean

/**
 * User OPT-IN persisted Android OS periodic public-market refresh.
 * JobScheduler may delay jobs under Doze and battery constraints. Uses
 * only Binance PUBLIC USD-M HTTPS GETs, no account API keys/POST/orders.
 *
 * A scheduled run fetches ONE of 15 pairs, all 15 native timeframes, one
 * bounded page each. Rotates symbols across subsequent runs and persists
 * progress in the same private SQLite used by the user-initiated download.
 * It NEVER enables trading or automatically starts the FGS after boot.
 */
class NativeWaveScheduledJob : JobService() {
    private val stop = AtomicBoolean(false)
    private var thread: Thread? = null

    override fun onStartJob(params: JobParameters): Boolean {
        stop.set(false)
        thread = Thread({
            var completed = false
            try {
                synchronized(NativeWaveConfig.marketSyncMutex) {
                    val settings = getSharedPreferences(PREFS, MODE_PRIVATE)
                    if (!settings.getBoolean(OPTED_IN, false)) return@Thread
                    val position = settings.getInt(NEXT_SYMBOL_INDEX, 0)
                        .coerceIn(0, NativeWaveConfig.symbols.lastIndex)
                    val symbol = NativeWaveConfig.symbols[position]
                    var terminalStatus = "Загрузка не завершена — причина неизвестна"
                    NativeWaveDownloader(applicationContext).onePass(
                        cancelled = { stop.get() || Thread.currentThread().isInterrupted },
                        onStatus = { message -> terminalStatus = message },
                        selectedSymbols = listOf(symbol),
                        pagesPerInterval = 1,
                        maxDurationMs = 4L * 60_000L
                    )
                    completed = terminalStatus.startsWith("Завершён ограниченный цикл:")
                    settings.edit().putString("scheduled_status",
                        if (completed) "Автоистория Binance PUBLIC $symbol: " +
                            terminalStatus.take(260)
                        else "Автозагрузка $symbol не завершена: " +
                            terminalStatus.take(260))
                        .putLong("scheduled_checked_ms", System.currentTimeMillis())
                        .apply()
                    if (completed && !stop.get()) {
                        settings.edit().putInt(
                            NEXT_SYMBOL_INDEX,
                            (position + 1) % NativeWaveConfig.symbols.size
                        ).apply()
                    }
                }
            } catch (error: InterruptedException) {
                Thread.currentThread().interrupt()
            } catch (error: Exception) {
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putString("scheduled_status",
                        "Автообновление не подтверждено: " +
                        error.javaClass.simpleName + ". Binance-ордера отключены.")
                    .putLong("scheduled_checked_ms", System.currentTimeMillis())
                    .apply()
            } finally {
                if (!stop.get()) jobFinished(params, false)
            }
        }, "WilliamsAI-OS-Public-Binance-Candles")
        thread?.start()
        return true
    }

    override fun onStopJob(params: JobParameters): Boolean {
        stop.set(true)
        thread?.interrupt()
        // This is a periodic job; the next scheduled opportunity can resume
        // from committed SQLite bars, not replay a speculative trading order.
        return false
    }

    companion object {
        private const val JOB_ID = 62009
        private const val PREFS = "native_market_public"
        private const val OPTED_IN = "scheduled_market_opt_in"
        private const val NEXT_SYMBOL_INDEX = "scheduled_pair_cursor"
        private const val HOUR_MS = 60L * 60L * 1_000L

        private fun scheduler(context: Context): JobScheduler =
            context.getSystemService(Context.JOB_SCHEDULER_SERVICE) as JobScheduler

        fun enabled(context: Context): Boolean {
            val opted = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getBoolean(OPTED_IN, false)
            return opted && scheduler(context).allPendingJobs.any { it.id == JOB_ID }
        }

        fun setEnabled(context: Context, enable: Boolean): Boolean {
            if (!enable) {
                context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                    .putBoolean(OPTED_IN, false).apply()
                scheduler(context).cancel(JOB_ID)
                return true
            }
            val job = JobInfo.Builder(
                JOB_ID, ComponentName(context, NativeWaveScheduledJob::class.java))
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
                .setPeriodic(HOUR_MS, 15L * 60L * 1_000L)
                .setPersisted(true) // Requires RECEIVE_BOOT_COMPLETED permission.
                .build()
            val accepted = scheduler(context).schedule(job) ==
                JobScheduler.RESULT_SUCCESS
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean(OPTED_IN, accepted).apply()
            return accepted
        }
    }
}
