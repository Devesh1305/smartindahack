package com.coiltag.app.data

import android.content.Context
import android.location.Location
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.coiltag.app.CoilTagApp.Companion.supabase
import com.coiltag.app.util.Images
import com.coiltag.app.util.Locator
import io.github.jan.supabase.auth.auth
import io.github.jan.supabase.exceptions.RestException
import io.github.jan.supabase.postgrest.from
import io.github.jan.supabase.postgrest.query.Order
import io.github.jan.supabase.storage.storage
import io.ktor.http.ContentType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.File
import java.time.Instant
import java.util.Date
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.time.Duration.Companion.hours

/**
 * All data access. Scans are first written to an on-device queue (photo + JSON),
 * then uploaded by [UploadWorker] whenever there is network - so scanning works
 * in yards with no signal and nothing is lost if the app is killed.
 */
object Repo {
    private const val TAG = "CoilTagRepo"
    const val BUCKET = "tag-photos"
    private val json = Json { ignoreUnknownKeys = true }
    private lateinit var app: Context
    private val uploadLock = Mutex()

    private val _pending = MutableStateFlow(0)
    val pendingCount: StateFlow<Int> = _pending

    private val queueDir get() = File(app.filesDir, "queue").apply { mkdirs() }

    fun init(ctx: Context) {
        app = ctx.applicationContext
        refreshPending()
        if (_pending.value > 0) scheduleUpload()
    }

    private fun refreshPending() {
        _pending.value = queueDir.listFiles { f -> f.extension == "json" }?.size ?: 0
    }

    // ---------------------------------------------------------------- auth/profile

    suspend fun myProfile(): Profile? {
        val uid = supabase.auth.currentUserOrNull()?.id ?: return null
        return supabase.from("profiles").select { filter { eq("id", uid) } }.decodeSingleOrNull<Profile>()
    }

    suspend fun allProfiles(): List<Profile> =
        supabase.from("profiles").select { order("created_at", Order.DESCENDING) }.decodeList()

    suspend fun setRole(userId: String, role: Role) {
        supabase.from("profiles").update({ set("role", role.name) }) { filter { eq("id", userId) } }
    }

    // ---------------------------------------------------------------- scans

    suspend fun scans(): List<Scan> =
        supabase.from("scans").select {
            order("captured_at", Order.DESCENDING)
            limit(5000)
        }.decodeList()

    suspend fun photoUrl(path: String): String =
        supabase.storage.from(BUCKET).createSignedUrl(path, 1.hours)

    /**
     * Geotags the full-resolution photo, keeps it in the gallery, and queues a
     * compressed geotagged copy + scan record for upload.
     */
    suspend fun queueScan(
        capture: File,
        batchId: String,
        manufacturer: String,
        product: String,
        fields: Map<String, String>,
        qrCodes: List<String>,
        loc: Location,
    ) = withContext(Dispatchers.IO) {
        val uid = supabase.auth.currentUserOrNull()?.id ?: error("Not signed in")
        val clientId = UUID.randomUUID().toString()
        val takenAt = Date()
        val caption = listOf(manufacturer, "Batch $batchId", product).filter { it.isNotBlank() }.joinToString(" | ")

        Images.geotag(capture, loc, caption, takenAt)
        runCatching { Images.saveToGallery(app, capture, "coil_${batchId}_${takenAt.time}.jpg") }
            .onFailure { Log.w(TAG, "gallery save failed", it) }

        val photo = File(queueDir, "$clientId.jpg")
        Images.writeUploadJpeg(Images.loadOriented(capture, 2048), photo)
        Images.geotag(photo, loc, caption, takenAt) // re-encoding dropped the EXIF

        val row = ScanInsert(
            clientId = clientId, batchId = batchId, manufacturer = manufacturer.ifBlank { null },
            product = product.ifBlank { null }, fields = fields + ("batch_id" to batchId), qrCodes = qrCodes,
            lat = loc.latitude, lon = loc.longitude, accuracyM = if (loc.hasAccuracy()) loc.accuracy else null,
            capturedAt = Instant.ofEpochMilli(loc.time.takeIf { it > 0 } ?: takenAt.time).toString(),
            imagePath = "$uid/$clientId.jpg", mockLocation = Locator.isMock(loc),
        )
        // JSON last: its presence marks the queue item as complete.
        File(queueDir, "$clientId.json").writeText(json.encodeToString(ScanInsert.serializer(), row))
        capture.delete()
        refreshPending()
        scheduleUpload()
    }

    fun scheduleUpload() {
        val req = OneTimeWorkRequestBuilder<UploadWorker>()
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()
        WorkManager.getInstance(app).enqueueUniqueWork("upload", ExistingWorkPolicy.APPEND_OR_REPLACE, req)
    }

    /** Uploads every queued scan. Safe to repeat: photo conflicts and duplicate client_ids are ignored. */
    suspend fun uploadPending(): Boolean = uploadLock.withLock {
        withContext(Dispatchers.IO) {
            var allOk = true
            val items = queueDir.listFiles { f -> f.extension == "json" }?.sortedBy { it.lastModified() } ?: emptyList()
            for (meta in items) {
                try {
                    val row = json.decodeFromString(ScanInsert.serializer(), meta.readText())
                    val photo = File(queueDir, meta.nameWithoutExtension + ".jpg")
                    if (photo.exists()) {
                        try {
                            supabase.storage.from(BUCKET).upload(row.imagePath, photo.readBytes()) {
                                upsert = false
                                contentType = ContentType.Image.JPEG
                            }
                        } catch (e: RestException) {
                            // Already uploaded by an earlier, interrupted attempt.
                            if (e.statusCode != 409 && !(e.message ?: "").contains("exists", true)) throw e
                        }
                    }
                    supabase.from("scans").upsert(row) {
                        onConflict = "client_id"
                        ignoreDuplicates = true
                    }
                    photo.delete()
                    meta.delete()
                } catch (e: Exception) {
                    Log.w(TAG, "upload failed for ${meta.name}", e)
                    allOk = false
                } finally {
                    refreshPending()
                }
            }
            allOk
        }
    }
}
