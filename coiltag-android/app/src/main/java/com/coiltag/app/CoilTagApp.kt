package com.coiltag.app

import android.app.Application
import com.coiltag.app.data.Repo
import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.auth.Auth
import io.github.jan.supabase.createSupabaseClient
import io.github.jan.supabase.postgrest.Postgrest
import io.github.jan.supabase.storage.Storage
import org.osmdroid.config.Configuration

class CoilTagApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // OpenStreetMap tile servers require an identifying user agent.
        Configuration.getInstance().userAgentValue = "$packageName/${BuildConfig.VERSION_NAME}"
        Repo.init(this)
    }

    companion object {
        /** Single client for the whole process (UI + background uploads share the login session). */
        val supabase: SupabaseClient by lazy {
            createSupabaseClient(BuildConfig.SUPABASE_URL, BuildConfig.SUPABASE_KEY) {
                install(Auth)
                install(Postgrest)
                install(Storage)
            }
        }
    }
}
