package com.coiltag.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.coiltag.app.data.Role

/** Admin-only: approve new sign-ups and set what each person may do. Enforced server-side by RLS. */
@Composable
fun UsersScreen(vm: AppViewModel) {
    LaunchedEffect(Unit) { vm.loadUsers() }
    val me = vm.profile?.id
    val pendingFirst = vm.users.sortedBy { if (it.role == Role.pending) 0 else 1 }
    LazyColumn(Modifier.fillMaxSize()) {
        item {
            Text("Pending users can't see anything. Viewers see the map; scanners also add scans; admins manage users.",
                Modifier.padding(16.dp), style = MaterialTheme.typography.bodySmall)
        }
        items(pendingFirst, key = { it.id }) { u ->
            Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp)) {
                Text(u.fullName?.ifBlank { null } ?: u.email ?: u.id, fontWeight = FontWeight.SemiBold)
                Text(u.email.orEmpty(), style = MaterialTheme.typography.bodySmall)
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Role.entries.forEach { r ->
                        FilterChip(
                            selected = u.role == r,
                            // You can't change your own role (prevents locking out the last admin).
                            enabled = u.id != me && !vm.busy,
                            onClick = { vm.setRole(u, r) },
                            label = { Text(r.name) },
                        )
                    }
                }
            }
            HorizontalDivider()
        }
    }
}
