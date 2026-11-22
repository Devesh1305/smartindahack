package com.coiltag.app.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.location.Location
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

object Images {

    /** Decode with EXIF rotation applied, longest side <= maxSide. */
    fun loadOriented(file: File, maxSide: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bounds)
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val raw = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: error("Could not decode photo")
        val degrees = when (ExifInterface(file.path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        val rotated = if (degrees == 0f) raw
        else Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(degrees) }, true)
        return scaleDown(rotated, maxSide)
    }

    fun scaleDown(bmp: Bitmap, maxSide: Int): Bitmap {
        val longest = max(bmp.width, bmp.height)
        if (longest <= maxSide) return bmp
        val s = maxSide.toFloat() / longest
        return Bitmap.createScaledBitmap(bmp, (bmp.width * s).roundToInt(), (bmp.height * s).roundToInt(), true)
    }

    /** ~150-250 KB JPEG that still keeps every line of a coil tag readable. */
    fun writeUploadJpeg(bmp: Bitmap, dst: File) {
        val small = scaleDown(bmp, 1280)
        dst.outputStream().use { small.compress(Bitmap.CompressFormat.JPEG, 82, it) }
        if (dst.length() > 480_000) { // storage bucket rejects > 500 KB
            dst.outputStream().use { small.compress(Bitmap.CompressFormat.JPEG, 65, it) }
        }
    }

    /** Writes GPS position, capture time and a "Manufacturer | Batch | Product" caption into the JPEG. */
    fun geotag(file: File, loc: Location, description: String, takenAt: Date) {
        val exif = ExifInterface(file.path)
        exif.setGpsInfo(loc)
        exif.setAttribute(ExifInterface.TAG_IMAGE_DESCRIPTION, description)
        exif.setAttribute(ExifInterface.TAG_USER_COMMENT, description)
        val stamp = SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Locale.US).format(takenAt)
        exif.setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, stamp)
        exif.setAttribute(ExifInterface.TAG_DATETIME, stamp)
        exif.saveAttributes()
    }

    /** Keeps the full-resolution geotagged original in the phone gallery (Pictures/CoilTag). */
    fun saveToGallery(ctx: Context, file: File, name: String) {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= 29) {
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/CoilTag")
            }
        }
        val uri = ctx.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        ctx.contentResolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
    }
}
