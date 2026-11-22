package com.coiltag.app.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.IconButton
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.produceState
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.coiltag.app.util.Images
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.coiltag.app.data.FIELD_LABELS
import com.coiltag.app.util.Locator
import java.io.File
import kotlin.math.roundToInt

private val Ok = Color(0xFF1A7F37)
private val Warn = Color(0xFF9A6700)

@Composable
fun ScanScreen(vm: AppViewModel) {
    val ctx = LocalContext.current
    var hasLocation by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
    }
    val permLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        hasLocation = it[Manifest.permission.ACCESS_FINE_LOCATION] == true
    }
    LaunchedEffect(Unit) {
        if (!hasLocation) permLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
    }
    // Keep GPS warm only while this screen is visible (battery).
    DisposableEffect(hasLocation) {
        if (hasLocation) vm.locator.start()
        onDispose { vm.locator.stop() }
    }

    var captureFile by rememberSaveable { mutableStateOf<String?>(null) }
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val f = captureFile?.let(::File)
        if (ok && f != null && f.length() > 0) vm.onPhotoTaken(f) else f?.delete()
    }
    val gallery = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            val f = vm.newCaptureFile()
            ctx.contentResolver.openInputStream(uri)?.use { input -> f.outputStream().use { input.copyTo(it) } }
            if (f.length() > 0) vm.onPhotoTaken(f) else f.delete()
        }
    }
    fun launchCamera() {
        val f = vm.newCaptureFile()
        captureFile = f.path
        camera.launch(FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", f))
    }

    when (val step = vm.step) {
        ScanStep.Idle -> IdleView(vm, hasLocation, onShoot = ::launchCamera, onGallery = { gallery.launch("image/*") },
            onGrant = { permLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)) })
        is ScanStep.Reading -> Busy("Reading QR codes and text…", step.photo)
        ScanStep.Saving -> Busy("Geotagging and saving…", null)
        is ScanStep.Review -> ReviewView(vm, step)
    }
}

@Composable
private fun GpsChip(vm: AppViewModel) {
    val loc by vm.locator.latest.collectAsState()
    val l = loc
    val (text, color) = when {
        l == null -> "GPS: searching…" to Warn
        Locator.isMock(l) -> "GPS: MOCK LOCATION" to MaterialTheme.colorScheme.error
        l.accuracy <= Locator.GOOD_ACCURACY_M -> "GPS ±${l.accuracy.roundToInt()} m" to Ok
        else -> "GPS weak ±${l.accuracy.roundToInt()} m" to Warn
    }
    AssistChip(onClick = {}, label = { Text(text, color = color) })
}

@Composable
private fun IdleView(vm: AppViewModel, hasLocation: Boolean, onShoot: () -> Unit, onGallery: () -> Unit, onGrant: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        if (hasLocation) GpsChip(vm) else {
            Text("Location permission is required to geotag scans.", color = MaterialTheme.colorScheme.error)
            TextButton(onClick = onGrant) { Text("Grant location") }
        }
        Spacer(Modifier.height(24.dp))
        Button(
            onClick = onShoot, enabled = hasLocation,
            modifier = Modifier.fillMaxWidth().height(96.dp), shape = RoundedCornerShape(16.dp),
        ) {
            Icon(Icons.Default.CameraAlt, null, Modifier.size(32.dp))
            Spacer(Modifier.size(12.dp))
            Text("Photograph coil tag", style = MaterialTheme.typography.titleLarge)
        }
        TextButton(onClick = onGallery, enabled = hasLocation) { Text("Use a photo from gallery") }
        Spacer(Modifier.height(8.dp))
        Text(
            "Fill the frame with the whole sticker, including every QR code. Avoid glare from the coil. Gallery photos are tagged with your current location, so scan them where the coil is.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun Busy(text: String, photo: File?) {
    Column(Modifier.fillMaxSize().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        photo?.let { PhotoPreview(it) }
        Spacer(Modifier.height(16.dp))
        CircularProgressIndicator()
        Spacer(Modifier.height(8.dp))
        Text(text)
    }
}

/**
 * Sharp preview of the tag photo (EXIF rotation applied). Tap to open full screen and pinch-zoom,
 * so the reviewer can compare the printed tag with the extracted fields.
 */
@Composable
private fun PhotoPreview(file: File) {
    val bmp by produceState<android.graphics.Bitmap?>(null, file) {
        value = withContext(Dispatchers.IO) { runCatching { Images.loadOriented(file, 1600) }.getOrNull() }
    }
    var fullScreen by remember { mutableStateOf(false) }
    Box(
        Modifier.fillMaxWidth().height(260.dp).clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .clickable(enabled = bmp != null) { fullScreen = true },
        contentAlignment = Alignment.Center,
    ) {
        val b = bmp
        if (b == null) CircularProgressIndicator()
        else {
            Image(b.asImageBitmap(), "Tag photo", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            Text("Tap to zoom", Modifier.align(Alignment.BottomEnd).padding(8.dp)
                .background(Color.Black.copy(alpha = 0.55f), RoundedCornerShape(6.dp)).padding(horizontal = 8.dp, vertical = 2.dp),
                color = Color.White, style = MaterialTheme.typography.labelSmall)
        }
    }
    if (fullScreen) ZoomablePhoto(file) { fullScreen = false }
}

@Composable
private fun ZoomablePhoto(file: File, onClose: () -> Unit) {
    val bmp by produceState<android.graphics.Bitmap?>(null, file) {
        value = withContext(Dispatchers.IO) { runCatching { Images.loadOriented(file, 2400) }.getOrNull() }
    }
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Box(Modifier.fillMaxSize().background(Color.Black)) {
            bmp?.let { b ->
                Image(
                    b.asImageBitmap(), "Tag photo",
                    Modifier.fillMaxSize()
                        .pointerInput(Unit) {
                            detectTransformGestures { _, pan, zoom, _ ->
                                scale = (scale * zoom).coerceIn(1f, 6f)
                                offset = if (scale == 1f) Offset.Zero else offset + pan
                            }
                        }
                        .graphicsLayer(scaleX = scale, scaleY = scale, translationX = offset.x, translationY = offset.y),
                    contentScale = ContentScale.Fit,
                )
            } ?: CircularProgressIndicator(Modifier.align(Alignment.Center))
            IconButton(onClick = onClose, modifier = Modifier.align(Alignment.TopStart).padding(top = 32.dp, start = 8.dp)) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = Color.White)
            }
        }
    }
}

@Composable
private fun ReviewView(vm: AppViewModel, step: ScanStep.Review) {
    val r = step.reading
    val loc = step.location
    var confirmWeak by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
        PhotoPreview(step.photo)
        r.warnings.forEach { w ->
            Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Warning, null, tint = Warn, modifier = Modifier.size(18.dp))
                Spacer(Modifier.size(6.dp))
                Text(w, color = Warn, style = MaterialTheme.typography.bodyMedium)
            }
        }
        Spacer(Modifier.height(8.dp))
        val fromQr = r.batchIdSource == "qr" || r.batchIdSource == "qci_url"
        OutlinedTextField(
            vm.batchId, { vm.batchId = it }, Modifier.fillMaxWidth(), singleLine = true,
            label = { Text("Batch ID") },
            supportingText = { Text(if (fromQr) "✓ read from QR code" else "⚠ not from QR - check it", color = if (fromQr) Ok else Warn) },
            trailingIcon = { if (fromQr) Icon(Icons.Default.CheckCircle, null, tint = Ok) },
        )
        OutlinedTextField(vm.manufacturer, { vm.manufacturer = it }, Modifier.fillMaxWidth(), label = { Text("Manufacturer") })
        OutlinedTextField(vm.product, { vm.product = it }, Modifier.fillMaxWidth(), label = { Text("Product") }, singleLine = true)
        val ordered = FIELD_LABELS.keys.filter { it in vm.fields } + vm.fields.keys.filter { it !in FIELD_LABELS }
        ordered.forEach { key ->
            OutlinedTextField(
                vm.fields[key].orEmpty(), { vm.fields[key] = it }, Modifier.fillMaxWidth(),
                label = { Text(FIELD_LABELS[key] ?: key) },
            )
        }

        if (r.qrCodes.isNotEmpty()) {
            Spacer(Modifier.height(12.dp))
            Text("QR codes found (${r.qrCodes.size})", fontWeight = FontWeight.SemiBold)
            r.qrCodes.forEach { Text(it, style = MaterialTheme.typography.bodySmall) }
        }

        Spacer(Modifier.height(12.dp))
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
            Column(Modifier.padding(12.dp)) {
                if (loc == null) Text("No GPS fix yet", color = MaterialTheme.colorScheme.error)
                else {
                    Text("📍 %.6f, %.6f".format(loc.latitude, loc.longitude))
                    Text("Accuracy ±${loc.accuracy.roundToInt()} m",
                        color = if (loc.accuracy <= Locator.GOOD_ACCURACY_M) Ok else Warn)
                    if (Locator.isMock(loc)) Text("Mock location app detected - this scan will be flagged.",
                        color = MaterialTheme.colorScheme.error)
                }
            }
        }
        Spacer(Modifier.height(16.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = { vm.retake() }, modifier = Modifier.weight(1f).height(52.dp)) { Text("Retake") }
            Button(
                onClick = { if (loc != null && loc.accuracy > Locator.GOOD_ACCURACY_M) confirmWeak = true else vm.save() },
                enabled = vm.batchId.isNotBlank(),
                modifier = Modifier.weight(2f).height(52.dp),
            ) { Text("Save geotagged scan") }
        }
        Spacer(Modifier.height(24.dp))
    }

    if (confirmWeak) AlertDialog(
        onDismissRequest = { confirmWeak = false },
        title = { Text("Weak GPS") },
        text = { Text("Accuracy is ±${loc?.accuracy?.roundToInt()} m. Move into open sky for a better fix, or save anyway.") },
        confirmButton = { TextButton(onClick = { confirmWeak = false; vm.save() }) { Text("Save anyway") } },
        dismissButton = { TextButton(onClick = { confirmWeak = false }) { Text("Wait") } },
    )
}

@Suppress("unused")
@Composable
private fun Fill(content: @Composable () -> Unit) = Box(Modifier.fillMaxSize()) { content() }
