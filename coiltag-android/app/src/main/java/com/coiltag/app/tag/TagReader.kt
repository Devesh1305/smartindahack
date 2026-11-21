package com.coiltag.app.tag

import android.graphics.Bitmap
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.tasks.await

/** On-device QR + OCR using ML Kit's bundled models (works with no network). */
object TagReader {
    private val textRecognizer by lazy { TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) }
    private val barcodeScanner by lazy {
        BarcodeScanning.getClient(
            BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE, Barcode.FORMAT_DATA_MATRIX).build()
        )
    }

    suspend fun read(bitmap: Bitmap): TagReading {
        val text = textRecognizer.process(InputImage.fromBitmap(bitmap, 0)).await()
        val boxes = text.textBlocks.flatMap { it.lines }.mapNotNull { line ->
            line.boundingBox?.let { TextBox(line.text, it.left.toFloat(), it.top.toFloat(), it.right.toFloat(), it.bottom.toFloat()) }
        }
        return TagParser.parse(boxes, decodeAll(bitmap))
    }

    /**
     * Full-frame pass, then overlapping half-size tiles upscaled 2x: tags often carry a
     * small secondary QR (e.g. the QCI "Made in India" code) that the full-frame pass misses.
     */
    private suspend fun decodeAll(bmp: Bitmap): List<String> {
        val found = LinkedHashSet<String>()
        suspend fun scan(b: Bitmap) {
            runCatching { barcodeScanner.process(InputImage.fromBitmap(b, 0)).await() }.getOrNull()
                ?.mapNotNull { it.rawValue?.trim() }?.filter { it.isNotEmpty() }?.let(found::addAll)
        }
        scan(bmp)
        val tw = bmp.width / 2
        val th = bmp.height / 2
        for (ty in 0..2) for (tx in 0..2) {
            val tile = Bitmap.createBitmap(bmp, tx * bmp.width / 4, ty * bmp.height / 4, tw, th)
            scan(Bitmap.createScaledBitmap(tile, tw * 2, th * 2, true))
        }
        return found.toList()
    }
}
