package com.example.llama

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context
import android.os.PersistableBundle

/**
 * User-initiated Android-native resilient market-data refresh.
 *
 * Native JobScheduler persists the PERIODIC job across device reboot, asks
 * OS for network access, and honors Doze/battery scheduling. It does not
 * promise exact 60-minute starts, bypass battery restrictions, run model
 * inference in background, or control positions. No foreground trading.
 */
class WaveMarketJobService : JobService() {
    @Volatile private var cancelled = false
    private var worker: Thread? = null

    override fun onStartJob(params: JobParameters): Boolean {
        cancelled = false
        val symbol = params.extras.getString(KEY_SYMBOL, "ETHUSDT")
        worker = Thread({
            try {
                synchronized(PUBLIC_READ_LOCK) {
                    check(symbol in WaveMarketResearch.SYMBOLS)
                    val results = WaveMarketResearch.updateAll(
                        applicationContext, symbol) {
                        cancelled || Thread.currentThread().isInterrupted
                    }
                    val observations = results.sumOf { it.newlyAdded }
                    val caught = results.count { it.state == "UP_TO_LATEST_AVAILABLE" }
                    getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                        .putString(KEY_RESULT,
                            "Binance PUBLIC USD-M $symbol: обновлено TF=" +
                            results.size + "/15, новые закрытые свечи=" +
                            observations + ", дошли до последней=" + caught +
                            "/15. Полнота архива с листинга НЕ доказана.")
                        .putLong(KEY_UPDATED, System.currentTimeMillis())
                        .apply()
                }
            } catch (error: Exception) {
                // No raw HTTP response URLs, credentials or trace to display.
                getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putString(KEY_RESULT, "Binance: загрузка НЕ подтверждена. " +
                        "Данные не подменяются, торговля отключена. Причина: " +
                        error.javaClass.simpleName + ", " +
                        (error.message ?: "сеть или формат").take(140))
                    .putLong(KEY_UPDATED, System.currentTimeMillis())
                    .apply()
            } finally {
                jobFinished(params, false)
            }
        }, "WilliamsAI-Public-Binance-Waves")
        worker?.start()
        return true
    }

    override fun onStopJob(params: JobParameters): Boolean {
        cancelled = true
        worker?.interrupt()
        // No secret and no orders; scheduler may resume on next PERIODIC job.
        return false
    }

    companion object {
        const val PREFS = "wave_market_readonly"
        const val KEY_SYMBOL = "symbol"
        const val KEY_RESULT = "last_result"
        const val KEY_UPDATED = "last_finished_ms"
        private const val PERIODIC_JOB_ID = 62001
        private const val MANUAL_JOB_ID = 62002
        private const val ONE_HOUR_MS = 60L * 60L * 1000L
        private val PUBLIC_READ_LOCK = Any()

        private fun scheduler(context: Context): JobScheduler =
            context.getSystemService(Context.JOB_SCHEDULER_SERVICE) as JobScheduler

        private fun makeJob(context: Context, id: Int, symbol: String): JobInfo.Builder {
            require(symbol in WaveMarketResearch.SYMBOLS)
            val extras = PersistableBundle()
            extras.putString(KEY_SYMBOL, symbol)
            return JobInfo.Builder(id,
                ComponentName(context, WaveMarketJobService::class.java))
                .setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
                .setExtras(extras)
        }

        fun enablePeriodic(context: Context, symbol: String): Boolean {
            require(symbol in WaveMarketResearch.SYMBOLS)
            val task = makeJob(context, PERIODIC_JOB_ID, symbol)
                .setPeriodic(ONE_HOUR_MS, 15L * 60L * 1000L)
                .setPersisted(true)
                .build()
            return scheduler(context).schedule(task) == JobScheduler.RESULT_SUCCESS
        }

        fun runOnce(context: Context, symbol: String): Boolean {
            require(symbol in WaveMarketResearch.SYMBOLS)
            val task = makeJob(context, MANUAL_JOB_ID, symbol)
                .setOverrideDeadline(0L)
                .build()
            return scheduler(context).schedule(task) == JobScheduler.RESULT_SUCCESS
        }

        fun disable(context: Context) {
            scheduler(context).cancel(PERIODIC_JOB_ID)
            scheduler(context).cancel(MANUAL_JOB_ID)
        }

        fun isEnabled(context: Context): Boolean =
            scheduler(context).allPendingJobs.any { it.id == PERIODIC_JOB_ID }
    }
}
