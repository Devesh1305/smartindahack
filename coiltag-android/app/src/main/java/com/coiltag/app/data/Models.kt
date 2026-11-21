package com.coiltag.app.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

enum class Role { pending, viewer, scanner, admin;
    val canScan get() = this == scanner || this == admin
    val canView get() = this != pending
}

@Serializable
data class Profile(
    val id: String,
    val email: String? = null,
    @SerialName("full_name") val fullName: String? = null,
    val role: Role = Role.pending,
)

/** What the phone sends. Server fills id, created_at and scanned_by (= the logged-in user). */
@Serializable
data class ScanInsert(
    @SerialName("client_id") val clientId: String,
    @SerialName("batch_id") val batchId: String,
    val manufacturer: String?,
    val product: String?,
    val fields: Map<String, String>,
    @SerialName("qr_codes") val qrCodes: List<String>,
    val lat: Double,
    val lon: Double,
    @SerialName("accuracy_m") val accuracyM: Float?,
    @SerialName("captured_at") val capturedAt: String,
    @SerialName("image_path") val imagePath: String,
    @SerialName("mock_location") val mockLocation: Boolean,
)

@Serializable
data class Scan(
    val id: String,
    @SerialName("batch_id") val batchId: String,
    val manufacturer: String? = null,
    val product: String? = null,
    val fields: Map<String, String> = emptyMap(),
    @SerialName("qr_codes") val qrCodes: List<String> = emptyList(),
    val lat: Double,
    val lon: Double,
    @SerialName("accuracy_m") val accuracyM: Float? = null,
    @SerialName("captured_at") val capturedAt: String,
    @SerialName("image_path") val imagePath: String? = null,
    @SerialName("mock_location") val mockLocation: Boolean = false,
    @SerialName("scanned_by") val scannedBy: String? = null,
)

/** Human labels for parsed field keys. */
val FIELD_LABELS = linkedMapOf(
    "size" to "Size", "heat_no" to "Heat No", "grade" to "Grade", "net_weight" to "Net weight",
    "gross_weight" to "Gross weight", "date" to "Mfg date", "pack_type" to "Pack type",
    "spec" to "Specification", "spec_2" to "Specification 2", "customer" to "Customer / Ship-to",
)
