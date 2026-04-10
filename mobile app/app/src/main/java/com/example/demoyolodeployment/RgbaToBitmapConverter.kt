package com.example.demoyolodeployment

import android.graphics.Bitmap
import android.graphics.PixelFormat
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

class RgbaToBitmapConverter {

    private var tightRgba: ByteArray? = null
    private var rowBuffer: ByteArray? = null

    // Reused wrapper to avoid ByteBuffer.wrap allocation each frame
    private var tightWrappedBuffer: ByteBuffer? = null

    fun rgbaToBitmap(image: ImageProxy, outBitmap: Bitmap) {
        require(image.format == PixelFormat.RGBA_8888) {
            "Expected RGBA_8888 ImageProxy. Ensure ImageAnalysis uses OUTPUT_IMAGE_FORMAT_RGBA_8888."
        }

        val w = image.width
        val h = image.height
        require(outBitmap.width == w && outBitmap.height == h) {
            "Output bitmap must match ImageProxy size. Bitmap=${outBitmap.width}x${outBitmap.height}, Image=${w}x${h}"
        }

        val plane = image.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        require(pixelStride == 4) { "Unexpected pixelStride=$pixelStride for RGBA_8888" }

        val tightSize = w * h * 4

        val tight = tightRgba?.takeIf { it.size == tightSize } ?: run {
            val arr = ByteArray(tightSize)
            tightRgba = arr
            tightWrappedBuffer = ByteBuffer.wrap(arr)
            arr
        }

        // If tightRgba existed but buffer wrapper was missing/mismatched, fix it
        val wrapped = tightWrappedBuffer?.takeIf { it.array() === tight && it.capacity() == tightSize } ?: run {
            val bb = ByteBuffer.wrap(tight)
            tightWrappedBuffer = bb
            bb
        }

        val expectedRowStride = w * 4

        buf.rewind()
        if (rowStride == expectedRowStride) {
            buf.get(tight, 0, tightSize)
        } else {
            val rowBuf = rowBuffer?.takeIf { it.size == rowStride }
                ?: ByteArray(rowStride).also { rowBuffer = it }

            var outPos = 0
            repeat(h) {
                buf.get(rowBuf, 0, rowStride)
                System.arraycopy(rowBuf, 0, tight, outPos, expectedRowStride)
                outPos += expectedRowStride
            }
        }

        wrapped.rewind()
        outBitmap.copyPixelsFromBuffer(wrapped)
    }
}