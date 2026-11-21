package com.coiltag.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import coil.compose.AsyncImage
import com.coiltag.app.data.FIELD_LABELS
import com.coiltag.app.data.Repo
import com.coiltag.app.data.Scan
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val TIME = DateTimeFormatter.ofPattern("dd MMM yyyy, HH:mm").withZone(ZoneId.systemDefault())
fun fmtTime(iso: String): String = runCatching { TIME.format(Instant.parse(iso)) }.getOrDefault(iso)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MapScreen(vm: AppViewModel) {
    LaunchedEffect(Unit) { vm.loadScans() }
    var query by remember { mutableStateOf("") }
    var selected by remember { mutableStateOf<Scan?>(null) }
    var fitted by remember { mutableStateOf(false) }

    val shown = vm.scans.filter {
        query.isBlank() || it.batchId.contains(query, true) || (it.manufacturer ?: "").contains(query, true)
    }
    val ctx = LocalContext.current
    val mapView = remember {
        MapView(ctx).apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            controller.setZoom(5.0)
            controller.setCenter(GeoPoint(21.0, 78.0))
        }
    }
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    DisposableEffect(lifecycle) {
        val obs = LifecycleEventObserver { _, e ->
            if (e == Lifecycle.Event.ON_RESUME) mapView.onResume()
            if (e == Lifecycle.Event.ON_PAUSE) mapView.onPause()
        }
        lifecycle.addObserver(obs)
        mapView.onResume()
        onDispose { lifecycle.removeObserver(obs); mapView.onPause() }
    }

    Column(Modifier.fillMaxSize()) {
        Row(Modifier.padding(horizontal = 12.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(query, { query = it }, Modifier.weight(1f), singleLine = true,
                placeholder = { Text("Filter batch / manufacturer") })
            IconButton(onClick = { vm.loadScans() }) {
                if (vm.busy) CircularProgressIndicator(Modifier.padding(8.dp)) else Icon(Icons.Default.Refresh, "Refresh")
            }
        }
        val coils = shown.groupBy { it.batchId }
        Text("${shown.size} scans · ${coils.size} coils", Modifier.padding(horizontal = 16.dp),
            style = MaterialTheme.typography.labelMedium)

        AndroidView(
            factory = { mapView },
            modifier = Modifier.fillMaxWidth().weight(1f),
            update = { map ->
                map.overlays.clear()
                val points = mutableListOf<GeoPoint>()
                coils.forEach { (batch, list) ->
                    // newest first: latest = current position; trail shows where the coil moved
                    if (list.size > 1) map.overlays.add(Polyline(map).apply {
                        setPoints(list.reversed().map { GeoPoint(it.lat, it.lon) })
                        outlinePaint.color = android.graphics.Color.rgb(200, 16, 46)
                        outlinePaint.strokeWidth = 5f
                    })
                    list.forEachIndexed { i, s ->
                        val p = GeoPoint(s.lat, s.lon)
                        points += p
                        map.overlays.add(Marker(map).apply {
                            position = p
                            title = "$batch · ${s.manufacturer ?: ""}"
                            alpha = if (i == 0) 1f else 0.45f
                            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                            setOnMarkerClickListener { _, _ -> selected = s; true }
                        })
                    }
                }
                if (!fitted && points.isNotEmpty()) {
                    fitted = true
                    map.post {
                        if (points.size == 1) { map.controller.setZoom(17.0); map.controller.setCenter(points[0]) }
                        else map.zoomToBoundingBox(BoundingBox.fromGeoPointsSafe(points).increaseByScale(1.3f), false)
                    }
                }
                map.invalidate()
            },
        )

        LazyColumn(Modifier.fillMaxWidth().height(200.dp)) {
            items(shown, key = { it.id }) { s ->
                Column(Modifier.fillMaxWidth().clickable {
                    selected = s
                    mapView.controller.animateTo(GeoPoint(s.lat, s.lon), 18.0, 600L)
                }.padding(horizontal = 16.dp, vertical = 8.dp)) {
                    Text(s.batchId, fontWeight = FontWeight.Bold)
                    Text("${s.manufacturer ?: "Unknown"} · ${s.product ?: ""}", style = MaterialTheme.typography.bodySmall)
                    Text("${fmtTime(s.capturedAt)} · %.5f, %.5f".format(s.lat, s.lon),
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                HorizontalDivider()
            }
        }
    }

    selected?.let { s ->
        ModalBottomSheet(onDismissRequest = { selected = null }) { ScanDetail(s) }
    }
}

@Composable
private fun ScanDetail(s: Scan) {
    var url by remember(s.id) { mutableStateOf<String?>(null) }
    LaunchedEffect(s.id) { url = s.imagePath?.let { runCatching { Repo.photoUrl(it) }.getOrNull() } }
    Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 32.dp)) {
        Text(s.manufacturer ?: "Unknown manufacturer", style = MaterialTheme.typography.titleMedium)
        Text("Batch ${s.batchId}" + (s.product?.let { " · $it" } ?: ""), style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold)
        Text("${fmtTime(s.capturedAt)} · ±${s.accuracyM?.toInt() ?: "?"} m", style = MaterialTheme.typography.bodySmall)
        if (s.mockLocation) Text("⚠ Location came from a mock-location app", color = MaterialTheme.colorScheme.error)
        Spacer(Modifier.height(8.dp))
        AsyncImage(url, "Tag photo", Modifier.fillMaxWidth().height(220.dp).clip(RoundedCornerShape(12.dp)),
            contentScale = ContentScale.Crop)
        Spacer(Modifier.height(8.dp))
        (FIELD_LABELS.keys.filter { it in s.fields } + s.fields.keys.filter { it !in FIELD_LABELS && it != "batch_id" })
            .forEach { k ->
                Row(Modifier.padding(vertical = 2.dp)) {
                    Text(FIELD_LABELS[k] ?: k, Modifier.weight(1f), color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(s.fields[k].orEmpty(), Modifier.weight(2f))
                }
            }
        Text("📍 %.6f, %.6f".format(s.lat, s.lon), Modifier.padding(top = 8.dp))
    }
}
