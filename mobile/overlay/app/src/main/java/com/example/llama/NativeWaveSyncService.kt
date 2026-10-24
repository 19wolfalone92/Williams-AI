package com.example.llama

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * User-initiated read-only PUBLIC USD-M download. Foreground dataSync, NOT
 * an account-facing trading service. Nothing is sent to Binance except GET.
 *
 * Android's FGS quotas/background start limits are respected by NOT
 * auto-launching on boot, by requiring a visible Activity action and by a
 * hard ~24 minute per-pass cap. Each tap resumes SQLite data; no imaginary
 * "runs forever after restart" promise.
 */
class NativeWaveSyncService : Service() {
    private val stop = AtomicBoolean(false)
    private var worker:Thread? = null
    private val channelId="williams_ai_market_public_download_v1"

    private fun updateState(value:String) {
        getSharedPreferences("native_market_public",MODE_PRIVATE).edit()
            .putString("summary",value.take(400))
            .putLong("updated_ms",System.currentTimeMillis()).apply()
        val notification=notice(value.take(110))
        val nm=getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(6221,notification)
    }

    private fun notice(message:String):Notification {
        val open=Intent(this,MainActivity::class.java)
        val openPending=PendingIntent.getActivity(this,11,open,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val stopIntent=Intent(this,NativeWaveSyncService::class.java)
            .setAction(ACTION_STOP)
        val stopPending=PendingIntent.getService(this,12,stopIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val builder=if(Build.VERSION.SDK_INT>=26)
            Notification.Builder(this,channelId) else Notification.Builder(this)
        return builder
            .setContentTitle("Williams AI — публичная история Binance")
            .setContentText(message)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(openPending)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_media_pause,
                "Остановить загрузку",stopPending)
            .build()
    }

    override fun onCreate() {
        super.onCreate()
        val manager=getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if(Build.VERSION.SDK_INT>=26)
            manager.createNotificationChannel(NotificationChannel(
                channelId,"История графиков Binance (только чтение)",
                NotificationManager.IMPORTANCE_LOW))
    }
    override fun onStartCommand(intent:Intent?,flags:Int,startId:Int):Int {
        if(intent?.action==ACTION_STOP){
            stop.set(true)
            worker?.interrupt()
            updateState("Загрузка остановлена пользователем. История сохранена.")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        if(intent?.action!=ACTION_START) {
            stopSelf()
            return START_NOT_STICKY
        }
        if(worker?.isAlive==true)return START_NOT_STICKY
        stop.set(false)
        if(Build.VERSION.SDK_INT>=29) {
            startForeground(6221,notice("Проверяю Binance USD-M..."),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else startForeground(6221,notice("Проверяю Binance USD-M..."))
        worker=Thread({
            try {
                NativeWaveDownloader(applicationContext).onePass(
                    cancelled={stop.get() || Thread.currentThread().isInterrupted},
                    onStatus={s->
                        if(!stop.get())updateState(s)
                    }
                )
            } catch(_:InterruptedException) {
                Thread.currentThread().interrupt()
            } catch(e:Exception) {
                updateState("Загрузка остановлена: ${e.message?.take(130) ?: "сбой сети или хранения"}. "+
                    "Ордеров нет, повтор выполняется только вручную.")
            } finally {
                if(!stop.get()) {
                    getSharedPreferences("native_market_public",MODE_PRIVATE).edit()
                        .putBoolean("running",false).apply()
                }
                stopSelf()
            }
        },"WilliamsAI-PUBLIC-BinanceHistory").also {
            getSharedPreferences("native_market_public",MODE_PRIVATE)
                .edit().putBoolean("running",true).apply()
            it.start()
        }
        // Android must NOT restart a lengthy HTTP downloader arbitrarily
        // after OS kill: next user tap safely resumes from SQLite.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stop.set(true)
        worker?.interrupt()
        getSharedPreferences("native_market_public",MODE_PRIVATE)
            .edit().putBoolean("running",false).apply()
        super.onDestroy()
    }
    override fun onBind(intent:Intent?):IBinder?=null

    companion object {
        const val ACTION_START="com.williamsai.offline.WAVE_PUBLIC_START"
        const val ACTION_STOP="com.williamsai.offline.WAVE_PUBLIC_STOP"
    }
}
