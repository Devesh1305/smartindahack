package com.coiltag.app

import android.app.Application
import com.coiltag.app.data.Repo
import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.auth.Auth
import io.github.jan.supabase.createSupabaseClient
import io.github.jan.supabase.functions.Functions
import io.github.jan.supabase.postgrest.Postgrest
import io.github.jan.supabase.storage.Storage
import org.osmdroid.config.Configuration

class CoilTagApp : Application() {
    override fun onCreate() {
        super.onCreate()
        installCrashRecorder()
        // A failure here would kill the app before any screen shows, so record it and carry on.
        try {
            // OpenStreetMap tile servers require an identifying user agent.
            Configuration.getInstance().userAgentValue = "$packageName/${BuildConfig.VERSION_NAME}"
            Repo.init(this)
        } catch (e: Throwable) {
            runCatching { crashFile(this).writeText("Start-up error\n\n" + e.stackTraceToString()) }
        }
    }

    /** Saves the stack trace of a crash so the next launch can show it (field phones have no logcat). */
    private fun installCrashRecorder() {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, e ->
            runCatching {
                crashFile(this).writeText(
                    "CoilTag ${BuildConfig.VERSION_NAME} · Android ${android.os.Build.VERSION.RELEASE} " +
                        "(SDK ${android.os.Build.VERSION.SDK_INT}) · ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}\n\n" +
                        e.stackTraceToString()
                )
            }
            previous?.uncaughtException(thread, e)
        }
    }

    companion object {
        fun crashFile(ctx: android.content.Context) = java.io.File(ctx.filesDir, "last_crash.txt")

        /** Single client for the whole process (UI + background uploads share the login session). */
        val supabase: SupabaseClient by lazy {
            createSupabaseClient(BuildConfig.SUPABASE_URL, BuildConfig.SUPABASE_KEY) {
                install(Auth)
                install(Postgrest)
                install(Storage)
                install(Functions)
            }
        }
    }
}
