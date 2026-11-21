# CoilTag for Android

The app photographs a steel coil tag, reads every QR code and the printed fields **on the phone**, and lets the user correct anything the OCR got wrong. It then saves the scan with the GPS position.

The geotagged photo and the data sync to Supabase. Every approved user sees the scans on a shared map.

## How it works

| Step | What happens |
|---|---|
| Photograph | The system camera takes a full-resolution photo. GPS is kept running while the Scan tab is open, so the fix is ready when the photo is taken. |
| Read | ML Kit (bundled models, **works offline**) decodes all QR codes in two passes: the full frame, then upscaled tiles, which catch small secondary QRs like the QCI "Made in India" code. It also OCRs the label. `TagParser` matches each caption to the value printed below it. The **QR batch ID is ground truth**, and a warning appears if the printed batch ID differs. |
| Review | The user confirms or edits the fields. GPS accuracy is shown; a fix worse than 50 m needs confirmation. Locations from **mock-GPS apps are flagged** on the scan. |
| Save | GPS, time and "Manufacturer \| Batch \| Product" are written into the photo's EXIF. The full-resolution photo goes to `Pictures/CoilTag` in the phone gallery. A compressed ~200 KB copy and the scan record go into an **on-device queue**. |
| Sync | WorkManager uploads the queue whenever there's network, retrying with backoff. Uploads are idempotent, so a scan is never duplicated even if a retry happens after a partial upload. |
| Map | OpenStreetMap shows one pin per coil. Earlier sightings of the same batch are faded and joined by a line showing how the coil moved. Tapping a pin shows the details and the photo. |

## Who can see what

This is enforced by Supabase Row Level Security on the server, not just in the app:

| Role | Map & scans | Add scans | Approve users |
|---|---|---|---|
| `pending` (every new sign-up) | ✗ | ✗ | ✗ |
| `viewer` | ✓ | ✗ | ✗ |
| `scanner` | ✓ | ✓ | ✗ |
| `admin` | ✓ | ✓ | ✓ (Users tab) |

Photos are in a **private** bucket and are served only through 1-hour signed URLs. Each user can upload only into their own folder.

## Accounts

There is **no public sign-up**. The admin creates every login from the app's **Users tab → Add person**:

- **Field staff without email** get a username such as `ramesh.k`. Internally it's stored as `ramesh.k@staff.coiltag.app`, but they only ever type the username.
- **Managers with company email** can use that email as their login instead.
- The app shows a **temporary password once**, with Copy and Share buttons. On first sign-in the person must set their own password (10+ characters).
- **Reset password** and **Remove person** are in each person's ⋮ menu. Removing someone keeps their past scans on the map.

Creating, resetting and removing accounts all run in the `admin-users` Supabase Edge Function. It holds the secret key, which never ships in the app, and re-checks that the caller is an admin on every request.

## One-time setup (Supabase dashboard)

1. **Authentication → Sign In / Providers: turn OFF "Allow new users to sign up".** Without this, anyone with the APK can create an account. That account would have no access, but it would still clutter the Users tab. Admin-created accounts keep working.
2. **Authentication → Sign In / Providers → Email:** set the minimum password length to 10.
3. **First admin** (already done for this project): `update public.profiles set role = 'admin' where email = '...';`

## Publishing privately on Google Play

Most field staff have no company Google account, so **Managed Google Play private apps can't reach their phones**: that channel only serves managed Workspace accounts. Use a **Play Console closed-testing track** as the permanent private channel instead:

1. Create a Play developer account (one-time fee) and a new app.
2. Set `coiltag.applicationId` in `gradle.properties` to your company's domain, e.g. `in.yourcompany.coiltag`. It **can never change** after the first upload.
3. Create the upload key (see `keystore.properties.example`), then run `./gradlew :app:bundleRelease`.
4. Upload `app/build/outputs/bundle/release/app-release.aab` to **Testing → Closed testing**. Add staff by the Google account (personal Gmail is fine) they use on their phone, ideally through a Google Group.
5. Under **App content**, fill in the privacy policy URL and the Data safety form (precise location, photos, name, email; encrypted in transit; deletable by the admin). Under **App access**, give the reviewer a `viewer` login.

Staff then install and get automatic updates from Play through the opt-in link. Managers with Workspace accounts can also get it as a Managed Google Play private app.

## Free-plan limits (current setup)

- **The project pauses after about 7 days with no use.** Restore it from the Supabase dashboard. Move to the Pro plan before relying on the app daily.
- **No automatic backups** on the free plan. Export scans regularly from Table Editor → scans → Export CSV.
- **1 GB of photo storage**, about 5,000 scans at ~200 KB each.

## Build

Requirements: JDK 17 or newer, and an Android SDK with platform 37.

```bash
./gradlew :app:assembleRelease          # phone APK (ARM only) for direct install
./gradlew :app:bundleRelease            # .aab for Google Play
./gradlew -Pemulator :app:assembleRelease  # also includes x86_64 for emulators
./gradlew :app:testDebugUnitTest        # parser tests against real tag photos
```

The output is `app/build/outputs/apk/release/app-release.apk`. To sideload it, enable "Install unknown apps" for your browser or file manager.

> Without `keystore.properties`, release builds are signed with the **debug key**. That's fine for direct installs, but Play rejects it.

## Database

The project is `coiltag` (ap-south-1). Its tables are `profiles` (role per user) and `scans` (batch, manufacturer, product, `fields` jsonb, `qr_codes`, lat/lon/accuracy, `captured_at`, `mock_location`, `image_path`). Photos are stored in the `tag-photos` bucket at `<user-id>/<client-id>.jpg`.
