package com.example.demoyolodeployment

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        textSize = 36f
    }

    private var detections: List<Detection> = emptyList()
    private var sx = 1f
    private var sy = 1f

    fun setDetections(list: List<Detection>, sx: Float, sy: Float) {
        detections = list
        this.sx = sx
        this.sy = sy
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (det in detections) {
            val b = det.box
            val scaled = RectF(
                b.left * sx,
                b.top * sy,
                b.right * sx,
                b.bottom * sy
            )
            canvas.drawRect(scaled, paint)
            canvas.drawText("${det.label} ${"%.2f".format(det.score)}",
                scaled.left, scaled.top - 10f, paint)
        }
    }
}