package com.coiltag.app.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Group
import androidx.compose.material.icons.filled.Key
import androidx.compose.material.icons.filled.Map
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.coiltag.app.data.Repo
import com.coiltag.app.data.Role
import io.github.jan.supabase.auth.status.SessionStatus

val Brand = Color(0xFFC8102E)

class MainActivity : ComponentActivity() {
    private val vm: AppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val crashFile = com.coiltag.app.CoilTagApp.crashFile(this)
        val lastCrash = crashFile.takeIf { it.exists() }?.readText()
        setContent {
            val dark = isSystemInDarkTheme()
            MaterialTheme(
                colorScheme = if (dark) darkColorScheme(primary = Color(0xFFFF6B7D), secondary = Color(0xFFFF6B7D))
                else lightColorScheme(primary = Brand, secondary = Brand)
            ) {
                var crash by remember { mutableStateOf(lastCrash) }
                val c = crash
                // Checked before touching the view model, so a crash in app start-up can still be reported.
                if (c != null) CrashReport(c) { crashFile.delete(); crash = null }
                else AppRoot(vm)
            }
        }
    }
}

@Composable
fun AppRoot(vm: AppViewModel) {
    val snackbar = remember { SnackbarHostState() }
    LaunchedEffect(vm.message) {
        vm.message?.let { snackbar.showSnackbar(it); vm.message = null }
    }
    val status by vm.session.collectAsState()
    Box(Modifier.fillMaxSize()) {
        when (status) {
            is SessionStatus.Initializing -> Centered { CircularProgressIndicator() }
            // RefreshFailure = we have a saved session but no network: keep working offline.
            is SessionStatus.Authenticated, is SessionStatus.RefreshFailure -> {
                LaunchedEffect(Unit) { vm.loadProfile() }
                val p = vm.profile
                when {
                    p == null -> Centered {
                        if (vm.busy) CircularProgressIndicator()
                        else Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text("Couldn't load your account.")
                            TextButton(onClick = { vm.loadProfile() }) { Text("Retry") }
                            TextButton(onClick = { vm.signOut() }) { Text("Sign out") }
                        }
                    }
                    p.mustChangePassword -> ChangePasswordScreen(vm, forced = true)
                    p.role == Role.pending -> PendingScreen(vm, p.login)
                    else -> HomeScreen(vm, p.role, snackbar)
                }
            }
            is SessionStatus.NotAuthenticated -> AuthScreen(vm)
        }
        SnackbarHost(snackbar, Modifier.align(Alignment.BottomCenter).padding(bottom = 96.dp))
    }
}

@Composable
fun CrashReport(trace: String, onDismiss: () -> Unit) {
    val clipboard = androidx.compose.ui.platform.LocalClipboardManager.current
    Column(Modifier.fillMaxSize().padding(16.dp).padding(top = 32.dp)) {
        Text("CoilTag crashed last time", style = MaterialTheme.typography.headlineSmall)
        Text("Copy this and send it to the developer.", style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(8.dp))
        androidx.compose.foundation.layout.Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { clipboard.setText(androidx.compose.ui.text.AnnotatedString(trace)) }) { Text("Copy error") }
            OutlinedButton(onClick = onDismiss) { Text("Continue") }
        }
        Spacer(Modifier.height(8.dp))
        Text(
            trace, Modifier.weight(1f).verticalScroll(rememberScrollState()),
            style = MaterialTheme.typography.bodySmall.copy(fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace),
        )
    }
}

@Composable
private fun Centered(content: @Composable () -> Unit) =
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { content() }

@Composable
fun AuthScreen(vm: AppViewModel) {
    var login by rememberSaveable { mutableStateOf("") }
    var password by rememberSaveable { mutableStateOf("") }
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(24.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("CoilTag", style = MaterialTheme.typography.displaySmall, color = MaterialTheme.colorScheme.primary)
        Text("Steel coil tag scanner", style = MaterialTheme.typography.bodyLarge)
        Spacer(Modifier.height(32.dp))
        OutlinedTextField(login, { login = it }, Modifier.fillMaxWidth(), label = { Text("Username or email") },
            singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email))
        OutlinedTextField(password, { password = it }, Modifier.fillMaxWidth(), label = { Text("Password") },
            singleLine = true, visualTransformation = PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password))
        Spacer(Modifier.height(16.dp))
        Button(
            onClick = { vm.signIn(login, password) },
            enabled = !vm.busy && login.isNotBlank() && password.isNotEmpty(),
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) { Text("Sign in") }
        Spacer(Modifier.height(16.dp))
        Text(
            "Accounts are created by your CoilTag admin. Ask them for your username and temporary password.",
            style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(),
        )
    }
}

const val MIN_PASSWORD = 10

@Composable
fun ChangePasswordScreen(vm: AppViewModel, forced: Boolean, onClose: () -> Unit = {}) {
    var pw by rememberSaveable { mutableStateOf("") }
    var pw2 by rememberSaveable { mutableStateOf("") }
    val tooShort = pw.isNotEmpty() && pw.length < MIN_PASSWORD
    val mismatch = pw2.isNotEmpty() && pw != pw2
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(24.dp), verticalArrangement = Arrangement.Center) {
        Text(if (forced) "Set your own password" else "Change password", style = MaterialTheme.typography.headlineSmall)
        if (forced) Text("You signed in with a temporary password from your admin. Choose a new one only you know.",
            style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(16.dp))
        OutlinedTextField(pw, { pw = it }, Modifier.fillMaxWidth(), label = { Text("New password") }, singleLine = true,
            visualTransformation = PasswordVisualTransformation(), isError = tooShort,
            supportingText = { Text("At least $MIN_PASSWORD characters") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password))
        OutlinedTextField(pw2, { pw2 = it }, Modifier.fillMaxWidth(), label = { Text("Repeat new password") }, singleLine = true,
            visualTransformation = PasswordVisualTransformation(), isError = mismatch,
            supportingText = { if (mismatch) Text("Passwords don't match") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password))
        Spacer(Modifier.height(16.dp))
        Button(
            onClick = { vm.changePassword(pw, onClose) },
            enabled = !vm.busy && pw.length >= MIN_PASSWORD && pw == pw2,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) { Text("Save password") }
        if (forced) TextButton(onClick = { vm.signOut() }, modifier = Modifier.fillMaxWidth()) { Text("Sign out") }
        else TextButton(onClick = onClose, modifier = Modifier.fillMaxWidth()) { Text("Cancel") }
    }
}

@Composable
fun PendingScreen(vm: AppViewModel, login: String?) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("No access yet", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(8.dp))
        Text("${login ?: "Your account"} is signed in, but it has no role. Ask your CoilTag admin to give you access.",
            textAlign = TextAlign.Center)
        Spacer(Modifier.height(24.dp))
        Button(onClick = { vm.loadProfile() }, enabled = !vm.busy) { Text("Check again") }
        OutlinedButton(onClick = { vm.signOut() }) { Text("Sign out") }
    }
}

private enum class Tab(val label: String) { Scan("Scan"), Map("Map"), Users("Users") }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(vm: AppViewModel, role: Role, snackbar: SnackbarHostState) {
    val tabs = buildList {
        if (role.canScan) add(Tab.Scan)
        add(Tab.Map)
        if (role == Role.admin) add(Tab.Users)
    }
    var tab by rememberSaveable { mutableStateOf(tabs.first()) }
    var changingPassword by rememberSaveable { mutableStateOf(false) }
    if (changingPassword) { ChangePasswordScreen(vm, forced = false) { changingPassword = false }; return }
    val pending by Repo.pendingCount.collectAsState()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("CoilTag · ${role.name}") },
                actions = {
                    if (pending > 0) IconButton(onClick = { Repo.scheduleUpload(); vm.message = "$pending scan(s) waiting for network" }) {
                        BadgedBox(badge = { Badge { Text("$pending") } }) { Icon(Icons.Default.CloudUpload, "Pending uploads") }
                    }
                    IconButton(onClick = { changingPassword = true }) { Icon(Icons.Default.Key, "Change password") }
                    IconButton(onClick = { vm.signOut() }) { Icon(Icons.AutoMirrored.Filled.Logout, "Sign out") }
                },
            )
        },
        bottomBar = {
            if (tabs.size > 1) NavigationBar {
                tabs.forEach { t ->
                    NavigationBarItem(
                        selected = tab == t, onClick = { tab = t }, label = { Text(t.label) },
                        icon = {
                            Icon(when (t) { Tab.Scan -> Icons.Default.CameraAlt; Tab.Map -> Icons.Default.Map; Tab.Users -> Icons.Default.Group }, t.label)
                        },
                    )
                }
            }
        },
    ) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            when (tab) {
                Tab.Scan -> ScanScreen(vm)
                Tab.Map -> MapScreen(vm)
                Tab.Users -> UsersScreen(vm)
            }
        }
    }
}
