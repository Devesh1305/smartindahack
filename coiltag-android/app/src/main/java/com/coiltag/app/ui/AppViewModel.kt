package com.coiltag.app.ui

import android.app.Application
import android.content.Context
import android.location.Location
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.coiltag.app.CoilTagApp.Companion.supabase
import com.coiltag.app.data.Profile
import com.coiltag.app.data.Repo
import com.coiltag.app.data.Role
import com.coiltag.app.data.Scan
import com.coiltag.app.tag.TagReader
import com.coiltag.app.tag.TagReading
import com.coiltag.app.util.Images
import com.coiltag.app.util.Locator
import io.github.jan.supabase.auth.auth
import io.github.jan.supabase.auth.providers.builtin.Email
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.File

sealed interface ScanStep {
    data object Idle : ScanStep
    data class Reading(val photo: File) : ScanStep
    data class Review(val photo: File, val reading: TagReading, val location: Location?) : ScanStep
    data object Saving : ScanStep
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    val session = supabase.auth.sessionStatus
    val locator = Locator(app)
    private val prefs = app.getSharedPreferences("coiltag", Context.MODE_PRIVATE)

    var profile by mutableStateOf<Profile?>(null); private set
    var busy by mutableStateOf(false); private set
    var message by mutableStateOf<String?>(null)

    var scans by mutableStateOf<List<Scan>>(emptyList()); private set
    var users by mutableStateOf<List<Profile>>(emptyList()); private set

    // --- scan flow (kept here so it survives recomposition)
    var step by mutableStateOf<ScanStep>(ScanStep.Idle); private set
    var batchId by mutableStateOf("")
    var manufacturer by mutableStateOf("")
    var product by mutableStateOf("")
    val fields = mutableStateMapOf<String, String>()

    private fun launchSafely(block: suspend () -> Unit) = viewModelScope.launch {
        busy = true
        try { block() } catch (e: Exception) { message = friendly(e) } finally { busy = false }
    }

    private fun friendly(e: Exception): String {
        val m = e.message ?: e.javaClass.simpleName
        return when {
            "Invalid login credentials" in m -> "Wrong email or password."
            "Email not confirmed" in m -> "Confirm your email first (check your inbox), then sign in."
            "Unable to resolve host" in m || "UnknownHost" in m -> "No internet connection."
            else -> m.lineSequence().first().take(200)
        }
    }

    // ---------------------------------------------------------------- auth

    fun signIn(email: String, password: String) = launchSafely {
        supabase.auth.signInWith(Email) { this.email = email.trim(); this.password = password }
    }

    fun signUp(name: String, email: String, password: String) = launchSafely {
        supabase.auth.signUpWith(Email) {
            this.email = email.trim(); this.password = password
            data = buildJsonObject { put("full_name", name.trim()) }
        }
        if (supabase.auth.currentSessionOrNull() == null) {
            message = "Account created. Confirm the link in your email, then sign in."
        }
    }

    fun signOut() = launchSafely {
        supabase.auth.signOut()
        profile = null
        prefs.edit().remove("role").apply()
    }

    /** Fetch role from server; fall back to the last known role when offline. */
    fun loadProfile() = viewModelScope.launch {
        busy = true
        try {
            profile = Repo.myProfile()?.also { prefs.edit().putString("role", it.role.name).apply() }
        } catch (e: Exception) {
            val cached = prefs.getString("role", null)?.let { runCatching { Role.valueOf(it) }.getOrNull() }
            val user = supabase.auth.currentUserOrNull()
            if (cached != null && user != null) {
                profile = Profile(user.id, user.email, null, cached)
                message = "Offline - scans will upload when you're back online."
            } else message = friendly(e)
        } finally { busy = false }
    }

    // ---------------------------------------------------------------- data

    fun loadScans() = launchSafely { scans = Repo.scans() }
    fun loadUsers() = launchSafely { users = Repo.allProfiles() }
    fun setRole(user: Profile, role: Role) = launchSafely {
        Repo.setRole(user.id, role)
        users = Repo.allProfiles()
    }

    // ---------------------------------------------------------------- scanning

    fun newCaptureFile(): File =
        File(getApplication<Application>().cacheDir, "captures").apply { mkdirs() }
            .let { File(it, "tag_${System.currentTimeMillis()}.jpg") }

    fun onPhotoTaken(photo: File) = viewModelScope.launch {
        step = ScanStep.Reading(photo)
        try {
            val loc = locator.fix() // position at the moment of the photo
            val reading = withContext(Dispatchers.Default) { TagReader.read(Images.loadOriented(photo, 2048)) }
            batchId = reading.batchId.orEmpty()
            manufacturer = reading.manufacturer.orEmpty()
            product = reading.product.orEmpty()
            fields.clear()
            reading.fields.filterKeys { it != "batch_id" }.forEach { (k, v) -> fields[k] = v }
            step = ScanStep.Review(photo, reading, loc)
        } catch (e: Exception) {
            message = "Could not read the tag: ${friendly(e)}"
            step = ScanStep.Idle
        }
    }

    fun retake() {
        (step as? ScanStep.Review)?.photo?.delete()
        step = ScanStep.Idle
    }

    fun save() {
        val review = step as? ScanStep.Review ?: return
        val loc = review.location ?: locator.latest.value
        if (loc == null) { message = "Waiting for a GPS fix - step outside and try again."; return }
        if (batchId.isBlank()) { message = "Batch ID is required."; return }
        step = ScanStep.Saving
        viewModelScope.launch {
            try {
                Repo.queueScan(review.photo, batchId.trim(), manufacturer.trim(), product.trim(),
                    fields.toMap(), review.reading.qrCodes, loc)
                message = "Saved ${batchId.trim()} - uploading in background"
                step = ScanStep.Idle
            } catch (e: Exception) {
                message = "Save failed: ${friendly(e)}"
                step = review
            }
        }
    }

    override fun onCleared() { locator.stop() }
}
