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
