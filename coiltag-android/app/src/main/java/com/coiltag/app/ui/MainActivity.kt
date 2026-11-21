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
        setContent {
            val dark = isSystemInDarkTheme()
            MaterialTheme(
                colorScheme = if (dark) darkColorScheme(primary = Color(0xFFFF6B7D), secondary = Color(0xFFFF6B7D))
                else lightColorScheme(primary = Brand, secondary = Brand)
            ) { AppRoot(vm) }
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
                    p.role == Role.pending -> PendingScreen(vm, p.email)
                    else -> HomeScreen(vm, p.role, snackbar)
                }
            }
            is SessionStatus.NotAuthenticated -> AuthScreen(vm)
        }
        SnackbarHost(snackbar, Modifier.align(Alignment.BottomCenter).padding(bottom = 96.dp))
    }
}

@Composable
private fun Centered(content: @Composable () -> Unit) =
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { content() }

@Composable
fun AuthScreen(vm: AppViewModel) {
    var signUp by rememberSaveable { mutableStateOf(false) }
    var name by rememberSaveable { mutableStateOf("") }
    var email by rememberSaveable { mutableStateOf("") }
    var password by rememberSaveable { mutableStateOf("") }
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(24.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("CoilTag", style = MaterialTheme.typography.displaySmall, color = MaterialTheme.colorScheme.primary)
        Text("Steel coil tag scanner", style = MaterialTheme.typography.bodyLarge)
        Spacer(Modifier.height(32.dp))
        if (signUp) OutlinedTextField(name, { name = it }, Modifier.fillMaxWidth(), label = { Text("Full name") }, singleLine = true)
        OutlinedTextField(email, { email = it }, Modifier.fillMaxWidth(), label = { Text("Work email") }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email))
        OutlinedTextField(password, { password = it }, Modifier.fillMaxWidth(), label = { Text("Password (min 6)") },
            singleLine = true, visualTransformation = PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password))
        Spacer(Modifier.height(16.dp))
        Button(
            onClick = { if (signUp) vm.signUp(name, email, password) else vm.signIn(email, password) },
            enabled = !vm.busy && email.isNotBlank() && password.length >= 6 && (!signUp || name.isNotBlank()),
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) { Text(if (signUp) "Create account" else "Sign in") }
        TextButton(onClick = { signUp = !signUp }, modifier = Modifier.fillMaxWidth()) {
            Text(if (signUp) "Have an account? Sign in" else "New here? Request access")
        }
        if (signUp) Text(
            "New accounts can't see or add anything until an admin approves them.",
            style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
fun PendingScreen(vm: AppViewModel, email: String?) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Waiting for approval", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(8.dp))
        Text("${email ?: "Your account"} is signed in, but an admin has to grant you access before you can scan or see the map.",
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
    val pending by Repo.pendingCount.collectAsState()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("CoilTag · ${role.name}") },
                actions = {
                    if (pending > 0) IconButton(onClick = { Repo.scheduleUpload(); vm.message = "$pending scan(s) waiting for network" }) {
                        BadgedBox(badge = { Badge { Text("$pending") } }) { Icon(Icons.Default.CloudUpload, "Pending uploads") }
                    }
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
