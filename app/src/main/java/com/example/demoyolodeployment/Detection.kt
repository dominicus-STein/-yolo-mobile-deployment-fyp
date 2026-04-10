package com.example.demoyolodeployment

import android.graphics.RectF
data class Detection(
    val box: RectF,      // left, top, right, bottom in input-image coords
    val label: String,   // class name
    val score: Float     // confidence 0.1
)