package com.coiltag.app.tag

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Fixtures are OCR boxes + QR payloads extracted from two real AM/NS India tag photos. */
class TagParserTest {

    private fun fixture(name: String): TagReading {
        val text = javaClass.classLoader!!.getResource("$name.json")!!.readText()
        val root = Json.parseToJsonElement(text).jsonObject
        val qr = root["qr"]!!.jsonArray.map { it.jsonPrimitive.content }
        val boxes = root["boxes"]!!.jsonArray.map {
            val o = it.jsonObject
            TextBox(
                o["text"]!!.jsonPrimitive.content,
                o["left"]!!.jsonPrimitive.float, o["top"]!!.jsonPrimitive.float,
                o["right"]!!.jsonPrimitive.float, o["bottom"]!!.jsonPrimitive.float,
            )
        }
        return TagParser.parse(boxes, qr)
    }

    @Test fun galvanizedTag() {
        val r = fixture("galvanized")
        assertEquals("9103KC000H", r.batchId)
        assertEquals("qr", r.batchIdSource)
        assertTrue(r.manufacturer!!.contains("AM/NS"))
        assertEquals("Galvanized", r.product)
        assertEquals("0.410 x 1220 x 5278", r.fields["size"])
        assertEquals("20.725 MT", r.fields["net_weight"])
        assertEquals("20.825 MT", r.fields["gross_weight"])
        assertEquals("11/10/2026 11:36:54 AM", r.fields["date"])
        assertEquals("DOM_OEM", r.fields["pack_type"])
        assertTrue(r.fields["spec"]!!.contains("GP250"))
        assertTrue(r.fields["customer"]!!.contains("PUNE"))
        assertTrue(r.warnings.toString(), r.warnings.isEmpty())
    }

    @Test fun hrCoilTagWithQciQr() {
        val r = fixture("hr_coil")
        assertEquals("NK9946", r.batchId)
        assertEquals("qr", r.batchIdSource)
        assertNotNull(r.qci)
        assertEquals("NK9946", r.qci!!.batchId)
        assertEquals("HR Coil", r.product)
        assertEquals("S40432", r.fields["heat_no"])
        assertEquals("ANCRMB2", r.fields["grade"])
        assertEquals("22.760 MT", r.fields["net_weight"])
        assertTrue(r.fields["size"]!!.startsWith("2.700 x 1260"))
        assertTrue(r.fields["spec"]!!.contains("11513"))
        assertTrue(r.warnings.toString(), r.warnings.isEmpty())
    }

    @Test fun ocrOnlyBatchIsFlagged() {
        val r = TagParser.parse(
            listOf(TextBox("Batch ID", 10f, 10f, 80f, 30f), TextBox("AB123", 10f, 40f, 80f, 60f)),
            emptyList(),
        )
        assertEquals("AB123", r.batchId)
        assertEquals("ocr", r.batchIdSource)
        assertTrue(r.warnings.any { "OCR only" in it })
    }

    @Test fun mismatchBetweenQrAndPrintIsFlagged() {
        val r = TagParser.parse(
            listOf(TextBox("Batch ID", 10f, 10f, 80f, 30f), TextBox("AB124", 10f, 40f, 80f, 60f)),
            listOf("AB123"),
        )
        assertEquals("AB123", r.batchId)
        assertTrue(r.warnings.any { "does not match" in it })
    }

    @Test fun printedZerosReadAsLettersAreNotAMismatch() {
        // Real failure from a phone: QR "9103KC000H", OCR of the print "9103KCOOOH".
        val r = TagParser.parse(
            listOf(TextBox("Batch ID", 10f, 10f, 80f, 30f), TextBox("9103KCOOOH", 10f, 40f, 120f, 60f)),
            listOf("9103KC000H"),
        )
        assertEquals("9103KC000H", r.batchId)
        assertTrue(r.warnings.toString(), r.warnings.none { "does not match" in it })
    }

    @Test fun realMismatchStillWarns() {
        val r = TagParser.parse(
            listOf(TextBox("Batch ID", 10f, 10f, 80f, 30f), TextBox("9103KC000J", 10f, 40f, 120f, 60f)),
            listOf("9103KC000H"),
        )
        assertTrue(r.warnings.any { "does not match" in it })
    }

    @Test fun underscoreSpacingAndNumericLookalikesAreCleaned() {
        assertEquals("DOM_OEM", TagParser.clean("pack_type", "DOM_ OEM"))
        assertEquals("IS 277_2018 GP250", TagParser.clean("spec", "IS 277_ 2018 GP250"))
        assertEquals("20.725 MT", TagParser.clean("net_weight", "2O.725MT"))
        assertEquals("0.410 x 1220 x 5278", TagParser.clean("size", "O.41O X l220 X 5278"))
        assertEquals("11/10/2026 11:36:54 AM", TagParser.clean("date", "11/1O/2026 11:36:54 AM"))
    }
}
