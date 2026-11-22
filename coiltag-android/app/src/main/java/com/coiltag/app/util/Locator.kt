package com.coiltag.app.util

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.os.Build
import android.os.Looper
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.tasks.await

/** Keeps a warm high-accuracy GPS fix while the scan screen is visible. */
class Locator(ctx: Context) {
    private val client = LocationServices.getFusedLocationProviderClient(ctx.applicationContext)
    private val _latest = MutableStateFlow<Location?>(null)
    val latest: StateFlow<Location?> = _latest
    private var running = false

    private val callback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation?.let { _latest.value = it }
        }
    }

    @SuppressLint("MissingPermission") // caller checks permission first
    fun start() {
        if (running) return
        running = true
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 2_000L).setMinUpdateIntervalMillis(1_000L).build()
        client.requestLocationUpdates(req, callback, Looper.getMainLooper())
    }

    fun stop() {
        if (!running) return
        running = false
        client.removeLocationUpdates(callback)
    }

    /** The warm fix if it is recent, otherwise ask for a fresh one. */
    @SuppressLint("MissingPermission")
    suspend fun fix(): Location? {
        val l = _latest.value
        if (l != null && System.currentTimeMillis() - l.time < 15_000) return l
        return runCatching { client.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null).await() }
            .getOrNull()?.also { _latest.value = it } ?: l
    }

    companion object {
        const val GOOD_ACCURACY_M = 50f
        fun isMock(l: Location): Boolean =
            if (Build.VERSION.SDK_INT >= 31) l.isMock else @Suppress("DEPRECATION") l.isFromMockProvider
    }
}
