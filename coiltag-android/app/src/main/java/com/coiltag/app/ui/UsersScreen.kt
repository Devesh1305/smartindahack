package com.coiltag.app.ui

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PersonAdd
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.coiltag.app.data.Profile
import com.coiltag.app.data.Role

/**
 * Admin-only: create logins for staff, reset passwords, set roles, remove people.
 * Account creation/reset/removal run in a server function that re-checks the caller is an admin;
 * roles are enforced server-side by Row Level Security.
 */
@Composable
fun UsersScreen(vm: AppViewModel) {
    LaunchedEffect(Unit) { vm.loadUsers() }
    val me = vm.profile?.id
    var adding by remember { mutableStateOf(false) }
    var removing by remember { mutableStateOf<Profile?>(null) }

    Box(Modifier.fillMaxSize()) {
        LazyColumn(Modifier.fillMaxSize()) {
            item {
                Text(
                    "viewer: sees the map · scanner: also adds scans · admin: also manages people · pending: no access",
                    Modifier.padding(16.dp), style = MaterialTheme.typography.bodySmall,
                )
            }
            items(vm.users.sortedBy { (it.fullName ?: it.login).lowercase() }, key = { it.id }) { u ->
                UserRow(u, isMe = u.id == me, busy = vm.busy,
                    onRole = { vm.setRole(u, it) }, onReset = { vm.resetPassword(u) }, onRemove = { removing = u })
                HorizontalDivider()
            }
            item { Spacer(Modifier.height(96.dp)) }
        }
        ExtendedFloatingActionButton(
            onClick = { adding = true },
            icon = { Icon(Icons.Default.PersonAdd, null) }, text = { Text("Add person") },
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
        )
    }

    if (adding) AddPersonDialog(vm) { adding = false }
    removing?.let { u ->
        AlertDialog(
            onDismissRequest = { removing = null },
            title = { Text("Remove ${u.fullName?.ifBlank { null } ?: u.login}?") },
            text = { Text("They will be signed out and can no longer log in. Scans they already made stay on the map.") },
            confirmButton = { TextButton(onClick = { vm.removeAccount(u); removing = null }) { Text("Remove") } },
            dismissButton = { TextButton(onClick = { removing = null }) { Text("Cancel") } },
        )
    }
    vm.issued?.let { (name, acc) -> CredentialsDialog(name, acc.login.orEmpty(), acc.tempPassword) { vm.issued = null } }
}

@Composable
private fun UserRow(u: Profile, isMe: Boolean, busy: Boolean, onRole: (Role) -> Unit, onReset: () -> Unit, onRemove: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(start = 16.dp, end = 4.dp, top = 10.dp, bottom = 10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text((u.fullName?.ifBlank { null } ?: u.login) + if (isMe) " (you)" else "", fontWeight = FontWeight.SemiBold)
                Text(u.login, style = MaterialTheme.typography.bodySmall)
            }
            if (!isMe) Box {
                IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, "More") }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(text = { Text("Reset password") }, onClick = { menu = false; onReset() })
                    DropdownMenuItem(text = { Text("Remove person") }, onClick = { menu = false; onRemove() })
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Role.entries.forEach { r ->
                FilterChip(
                    selected = u.role == r,
                    // You can't change your own role (prevents locking out the last admin).
                    enabled = !isMe && !busy,
                    onClick = { onRole(r) },
                    label = { Text(r.name) },
                )
            }
        }
    }
}

@Composable
private fun AddPersonDialog(vm: AppViewModel, onClose: () -> Unit) {
    var name by remember { mutableStateOf("") }
    var login by remember { mutableStateOf("") }
    var role by remember { mutableStateOf(Role.scanner) }
    val validLogin = if ('@' in login) Regex("""^[^@\s]+@[^@\s]+\.[^@\s]+$""").matches(login.trim())
    else Regex("""^[a-zA-Z0-9][a-zA-Z0-9._-]{2,30}$""").matches(login.trim())
    AlertDialog(
        onDismissRequest = onClose,
        title = { Text("Add person") },
        text = {
            Column {
                OutlinedTextField(name, { name = it }, Modifier.fillMaxWidth(), label = { Text("Full name") }, singleLine = true)
                OutlinedTextField(login, { login = it }, Modifier.fillMaxWidth(), singleLine = true,
                    label = { Text("Username or company email") },
                    supportingText = { Text("No email? Use a username like ramesh.k") },
                    isError = login.isNotBlank() && !validLogin)
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(Role.viewer, Role.scanner, Role.admin).forEach { r ->
                        FilterChip(selected = role == r, onClick = { role = r }, label = { Text(r.name) })
                    }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = !vm.busy && name.isNotBlank() && validLogin,
                onClick = { vm.createAccount(name, login, role, onClose) }) { Text("Create login") }
        },
        dismissButton = { TextButton(onClick = onClose) { Text("Cancel") } },
    )
}

/** Shown once: the temporary password is not stored anywhere the admin can see later. */
@Composable
private fun CredentialsDialog(name: String, login: String, password: String, onClose: () -> Unit) {
    val ctx = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val message = "CoilTag login for $name\nUsername: $login\nTemporary password: $password\n" +
        "You'll be asked to set your own password when you first sign in."
    AlertDialog(
        onDismissRequest = {},
        title = { Text("Login for $name") },
        text = {
            Column {
                Text("Give these to $name. The password is shown only now; if it's lost, use Reset password.")
                Spacer(Modifier.height(12.dp))
                SelectionContainer {
                    Column {
                        Text("Username: $login", fontFamily = FontFamily.Monospace)
                        Text("Password: $password", fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                ctx.startActivity(Intent.createChooser(
                    Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, message), "Send login"))
            }) { Text("Share") }
        },
        dismissButton = {
            Row {
                TextButton(onClick = { clipboard.setText(AnnotatedString(message)) }) { Text("Copy") }
                TextButton(onClick = onClose) { Text("Done") }
            }
        },
    )
}
