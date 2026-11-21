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

## One-time setup (Supabase dashboard)

1. **Authentication → Sign In / Providers → Email: turn off "Confirm email".** Supabase's built-in email service only sends to members of your Supabase organization, so field staff would never get a confirmation link. Admin approval already controls access.
2. **Make the first admin.** Sign up in the app, then run this in the SQL editor:
   ```sql
   update public.profiles set role = 'admin' where email = 'you@company.com';
   ```
   After that, approve everyone else from the app's **Users** tab.

## Build

Requirements: JDK 17 or newer, and an Android SDK with platform 37.

```bash
./gradlew :app:assembleRelease          # phone APK (ARM only)
./gradlew -Pemulator :app:assembleRelease  # also includes x86_64 for emulators
./gradlew :app:testDebugUnitTest        # parser tests against real tag photos
```

The output is `app/build/outputs/apk/release/app-release.apk`. To sideload it, enable "Install unknown apps" for your browser or file manager.

> The release build is signed with the **debug key** so it installs directly. Create a real keystore before any Play Store release.

## Database

The project is `coiltag` (ap-south-1). Its tables are `profiles` (role per user) and `scans` (batch, manufacturer, product, `fields` jsonb, `qr_codes`, lat/lon/accuracy, `captured_at`, `mock_location`, `image_path`). Photos are stored in the `tag-photos` bucket at `<user-id>/<client-id>.jpg`.
