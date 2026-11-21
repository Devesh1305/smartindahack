# Keep line numbers for readable crash reports.
-keepattributes SourceFile,LineNumberTable,Signature,InnerClasses,EnclosingMethod,*Annotation*

# App data models are (de)serialised by kotlinx.serialization.
-keep class com.coiltag.app.data.** { *; }

# Ktor finds its HTTP engine via ServiceLoader; supabase-kt plugins are looked up by type.
-keep class io.ktor.** { *; }
-keep class io.github.jan.supabase.** { *; }
-keep class kotlinx.serialization.** { *; }

# WorkManager instantiates workers by class name.
-keep class * extends androidx.work.ListenableWorker { <init>(...); }

-dontwarn org.slf4j.**
-dontwarn io.ktor.**
-dontwarn org.osmdroid.**
-dontwarn java.lang.management.**

# WorkManager's Room database (WorkDatabase_Impl) is created by reflection at start-up.
# Without this R8 strips its constructor and the app crashes before the first screen.
-keep class * extends androidx.room.RoomDatabase { <init>(...); }

# ML Kit's native OCR/barcode pipelines read Java fields over JNI by name.
# R8 full mode (AGP 9 default) renames them and OCR fails with a NullPointerException.
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_** { *; }
-keep class com.google.android.libraries.barhopper.** { *; }
-keep class com.google.photos.vision.barhopper.** { *; }

# Shrink but don't rename: field crash reports stay readable.
-dontobfuscate
