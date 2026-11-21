package com.coiltag.app.tag

import kotlin.math.abs
import kotlin.math.max

/** One line of recognised text with its bounding box (image pixels). */
data class TextBox(val text: String, val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val height: Float get() = max(bottom - top, 1f)
}

data class QciInfo(val url: String, val productId: String, val batchId: String, val serial: String)

data class TagReading(
    val qrCodes: List<String>,
    val batchId: String?,
    /** "qr", "qci_url", "ocr" or null */
    val batchIdSource: String?,
    val manufacturer: String?,
    val product: String?,
    /** Canonical field name -> value, in label order. */
    val fields: LinkedHashMap<String, String>,
    val warnings: List<String>,
    val ocrLines: List<String>,
    val qci: QciInfo?,
)

/**
 * Turns OCR lines + decoded QR payloads into structured tag data.
 *
 * Real AM/NS India tags put only the batch ID (and sometimes a QCI "Made in India"
 * URL) in their QR codes; everything else is printed text, laid out as a caption
 * with its value directly underneath. The QR batch ID is treated as ground truth.
 */
object TagParser {

    /** Normalised caption -> canonical field. OCR drops spaces/punctuation, so keys are alphanumeric only. */
    private val LABELS = mapOf(
        "batchid" to "batch_id", "batch" to "batch_id", "batchno" to "batch_id", "coilno" to "batch_id",
        "heatno" to "heat_no", "heat" to "heat_no",
        "size" to "size", "sizetmmxwmmxlmtr" to "size", "sizetmmxwmm" to "size",
        "grade" to "grade",
        "weight" to "net_weight", "netwt" to "net_weight", "netweight" to "net_weight",
        "grosswt" to "gross_weight", "grossweight" to "gross_weight",
        "date" to "date",
        "packtype" to "pack_type",
        "specification" to "spec", "spec" to "spec", "spec1" to "spec", "spect" to "spec",
        "specl" to "spec", "speci" to "spec",
        "customer" to "customer", "customershiptoparty" to "customer", "shiptoparty" to "customer",
    )
    private val MULTILINE = setOf("customer")

    private val MANUFACTURERS = listOf(
        "amns" to "ArcelorMittal Nippon Steel India (AM/NS India)",
        "arcelormittalnippon" to "ArcelorMittal Nippon Steel India (AM/NS India)",
        "tatasteel" to "Tata Steel",
        "jswsteel" to "JSW Steel",
        "jindalsteel" to "Jindal Steel & Power",
        "jspl" to "Jindal Steel & Power",
        "steelauthorityofindia" to "Steel Authority of India (SAIL)",
        "essarsteel" to "Essar Steel",
        "rinl" to "RINL / Vizag Steel",
    )
    private val PRODUCTS = listOf(
        "GALVANIZED" to "Galvanized", "GALVALUME" to "Galvalume", "HR COIL" to "HR Coil",
        "CR COIL" to "CR Coil", "HRPO" to "HRPO", "PPGI" to "PPGI", "COLOR COATED" to "Color Coated",
        "TINPLATE" to "Tinplate",
    )

    private val QCI_URL = Regex("""qcin\.org/product-details/([0-9a-fA-F]+)/([^/]+)/(\d+)""")

    fun norm(s: String): String = s.lowercase().filter { it in 'a'..'z' || it in '0'..'9' }

    private fun labelKey(b: TextBox): String? = LABELS[norm(b.text)]

    /** Values sit directly under their caption, left-aligned with it. */
    private fun valuesBelow(label: TextBox, boxes: List<TextBox>, multiline: Boolean): List<TextBox> {
        val out = mutableListOf<TextBox>()
        var anchor = label
        while (true) {
            val next = boxes.filter { b ->
                b !== anchor && b !in out &&
                    b.top > anchor.top + 0.5f * anchor.height &&
                    b.top < anchor.bottom + 1.8f * label.height &&
                    abs(b.left - label.left) < 3f * label.height
            }.minByOrNull { it.top } ?: break
            if (labelKey(next) != null) break // reached the next caption
            out += next
            if (!multiline || out.size >= 3) break
            anchor = next
        }
        return out
    }

    /**
     * Characters OCR cannot tell apart in label fonts (a printed 0 and O are identical).
     * Used only for comparing, never to rewrite text that might genuinely contain letters.
     */
    private val LOOKALIKE = mapOf('O' to '0', 'Q' to '0', 'D' to '0', 'I' to '1', 'L' to '1', '|' to '1',
        'S' to '5', 'B' to '8', 'Z' to '2', 'G' to '6')

    fun lookalikeKey(s: String): String =
        s.uppercase().filter { it.isLetterOrDigit() || it == '|' }.map { LOOKALIKE[it] ?: it }.joinToString("")

    /** In fields that only ever hold numbers, OCR letters that look like digits are digits. */
    private fun digitsOnly(v: String): String {
        val chars = v.toCharArray()
        for (i in chars.indices) {
            val c = chars[i]
            if (c !in "OoIl|") continue
            val prev = chars.getOrNull(i - 1)
            val next = chars.getOrNull(i + 1)
            // only when touching a digit or decimal point, so "MT" / "AM" units are left alone
            if ((prev != null && (prev.isDigit() || prev == '.' || prev == '/' || prev == ':')) ||
                (next != null && (next.isDigit() || next == '.' || next == '/' || next == ':'))
            ) chars[i] = if (c == 'O' || c == 'o') '0' else '1'
        }
        return String(chars)
    }

    fun clean(key: String, raw: String): String {
        // OCR adds a space after underscores: "DOM_ OEM" -> "DOM_OEM"
        var v = raw.trim().replace(Regex("""\s*_\s*"""), "_")
        if (key in setOf("size", "net_weight", "gross_weight", "date")) v = digitsOnly(v)
        when (key) {
            "size" -> v = v.replace(Regex("""\s*[xX×]\s*"""), " x ").replace('（', '(').replace('）', ')')
            "net_weight", "gross_weight" -> Regex("""\d+(?:\.\d+)?""").find(v.replace(',', '.'))?.let {
                val unit = if (v.uppercase().contains("MT") || it.value.toDouble() < 100) " MT" else ""
                v = it.value + unit
            }
            "date" -> Regex("""(\d{1,2}[/-]\d{1,2}[/-]\d{4})\s*(\d{1,2}:\d{2}(?::\d{2})?)?\s*([AaPp][Mm])?""")
                .find(v)?.let { m -> v = m.groupValues.drop(1).filter { it.isNotEmpty() }.joinToString(" ") }
            "spec" -> v = v.replace(Regex("""^[1l|]S"""), "IS") // OCR often reads "IS" as "1S"
        }
        return v
    }

    fun parse(rawBoxes: List<TextBox>, qrCodes: List<String>): TagReading {
        val boxes = rawBoxes.filter { it.text.isNotBlank() }.sortedWith(compareBy({ it.top }, { it.left }))
        val warnings = mutableListOf<String>()
        val full = norm(boxes.joinToString(" ") { it.text })
        val upper = boxes.joinToString(" ") { it.text.uppercase() }

        val manufacturer = MANUFACTURERS.firstOrNull { full.contains(it.first) }?.second
        val product = PRODUCTS.firstOrNull { (needle, _) ->
            upper.contains(needle) || upper.replace(" ", "").contains(needle.replace(" ", ""))
        }?.second

        val fields = LinkedHashMap<String, String>()
        for (b in boxes) {
            val key = labelKey(b) ?: continue
            val values = valuesBelow(b, boxes, key in MULTILINE)
            if (values.isEmpty()) continue
            val value = clean(key, values.joinToString(" / ") { it.text })
            var k = key
            var i = 2
            while (k in fields) k = "${key}_${i++}" // e.g. two "Spec." captions on HR tags
            fields[k] = value
        }

        val qrs = qrCodes.map { it.trim() }.filter { it.isNotEmpty() }.distinct()
        val plain = qrs.filterNot { it.startsWith("http", ignoreCase = true) }
        val qci = qrs.firstNotNullOfOrNull { q ->
            QCI_URL.find(q)?.let { QciInfo(q, it.groupValues[1], it.groupValues[2], it.groupValues[3]) }
        }
        val ocrBatch = fields["batch_id"]
        val (batch, source) = when {
            plain.isNotEmpty() -> plain.first() to "qr"
            qci != null -> qci.batchId to "qci_url"
            ocrBatch != null -> {
                warnings += "No QR decoded - batch ID comes from OCR only; verify it."
                ocrBatch to "ocr"
            }
            else -> {
                warnings += "No batch ID found. Retake the photo closer to the tag."
                null to null
            }
        }
        // Compare ignoring look-alikes: OCR reads a printed "000" as "OOO", but the QR is exact.
        if (batch != null && ocrBatch != null && lookalikeKey(ocrBatch) != lookalikeKey(batch)) {
            warnings += "Printed batch '$ocrBatch' does not match QR '$batch'."
        }
        fields["batch_id"] = batch ?: ""
        if (manufacturer == null) warnings += "Manufacturer not recognised - enter it manually."

        return TagReading(qrs, batch, source, manufacturer, product, fields, warnings, boxes.map { it.text }, qci)
    }
}
