package com.coiltag.app.data

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.coiltag.app.CoilTagApp.Companion.supabase
import io.github.jan.supabase.auth.auth

class UploadWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result {
        supabase.auth.awaitInitialization() // restores the saved login session
        if (supabase.auth.currentSessionOrNull() == null) return Result.retry()
        return if (Repo.uploadPending()) Result.success() else Result.retry()
    }
}
