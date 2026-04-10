package com.example.demoyolodeployment

import android.content.Context
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    private val context: Context,
    private val modelAssetName: String
) {
    private val TAG = "YoloDebug"

    // Guard all interpreter access (run/close) with one lock
    private val interpreterLock = Any()

    private var interpreter: Interpreter = createInterpreter(modelAssetName)

    // Input shape
    val inputWidth: Int
    val inputHeight: Int
    val isNCHW: Boolean   // true if model expects [1, 3, H, W]

    // Input dtype (sanity logging)
    private val inputType: DataType

    // Output parsing
    private val numDetections: Int
    private val numClasses: Int
    private val channelsLast: Boolean  // true: [1, numDet, features]; false: [1, features, numDet]

    // Layout
    private val hasObj: Boolean
    private val clsOffset: Int
    private val featuresPerDet: Int

    // COCO labels (still loaded), but we may not use them 1:1 for class mapping
    private val cocoLabels: List<String>

    // Effective class names for THIS model output (size = numClasses)
    private val classNames: List<String>

    // Reusable buffers
    private val inputByteBuffer: ByteBuffer
    private val outChannelsLast: Array<Array<FloatArray>>?
    private val outChannelsFirst: Array<Array<FloatArray>>?

    data class TimedDetections(
        val detections: List<Detection>,
        val inferenceMs: Double,
        // Max score for target label across all candidates, even if below threshold
        val maxTargetScore: Double,
        // NEW: max score across ANY class across all candidates
        val maxAnyScore: Double,
        // NEW: label of the maxAnyScore (empty if unknown)
        val maxAnyLabel: String
    )

    init {
        val inTensor = interpreter.getInputTensor(0)
        val inShape = inTensor.shape()
        inputType = inTensor.dataType()
        Log.d(TAG, "Model=$modelAssetName Input shape=${inShape.contentToString()}, type=$inputType")

        val tmpWidth: Int
        val tmpHeight: Int
        val tmpIsNCHW: Boolean

        if (inShape.size == 4 && inShape[3] == 3) {
            // [1, H, W, 3]
            tmpHeight = inShape[1]
            tmpWidth = inShape[2]
            tmpIsNCHW = false
        } else if (inShape.size == 4 && inShape[1] == 3) {
            // [1, 3, H, W]
            tmpHeight = inShape[2]
            tmpWidth = inShape[3]
            tmpIsNCHW = true
        } else {
            throw IllegalStateException("Unexpected input shape: ${inShape.contentToString()}")
        }

        inputWidth = tmpWidth
        inputHeight = tmpHeight
        isNCHW = tmpIsNCHW

        // Load COCO-80 labels (as you want to keep them)
        cocoLabels = FileUtil.loadLabels(context, "labels.txt")

        val outTensor = interpreter.getOutputTensor(0)
        val outShape = outTensor.shape()
        Log.d(TAG, "Output[0] shape=${outShape.contentToString()} type=${outTensor.dataType()}")

        if (outShape.size != 3) {
            throw IllegalStateException("Unexpected output[0] rank: ${outShape.contentToString()}")
        }

        val dim1 = outShape[1]
        val dim2 = outShape[2]

        val features = min(dim1, dim2)
        val detections = max(dim1, dim2)

        if (features <= 4) {
            throw IllegalStateException("features dim <= 4: $features in shape=${outShape.contentToString()}")
        }

        numDetections = detections
        channelsLast = (dim1 == numDetections)

        // Robust inference of layout/classes WITHOUT relying on labels.txt count matching.
        val tmpHasObj: Boolean
        val tmpClsOffset: Int
        val tmpNumClasses: Int

        when {
            // Match COCO-80 exactly (no obj): 4+80=84
            features == 4 + cocoLabels.size -> {
                tmpHasObj = false
                tmpClsOffset = 4
                tmpNumClasses = cocoLabels.size
            }
            // Match COCO-80 exactly (with obj): 5+80=85
            features == 5 + cocoLabels.size -> {
                tmpHasObj = true
                tmpClsOffset = 5
                tmpNumClasses = cocoLabels.size
            }
            // Single-class (no obj): 4+1=5
            features == 5 -> {
                tmpHasObj = false
                tmpClsOffset = 4
                tmpNumClasses = 1
            }
            // Single-class (with obj): 5+1=6
            features == 6 -> {
                tmpHasObj = true
                tmpClsOffset = 5
                tmpNumClasses = 1
            }
            // Fallback: assume obj exists and remaining are classes
            features > 5 -> {
                tmpHasObj = true
                tmpClsOffset = 5
                tmpNumClasses = features - 5
                Log.w(TAG, "features=$features not matching known patterns; fallback hasObj=true numClasses=$tmpNumClasses")
            }
            else -> {
                tmpHasObj = false
                tmpClsOffset = 4
                tmpNumClasses = features - 4
                Log.w(TAG, "features=$features unusual; fallback hasObj=false numClasses=$tmpNumClasses")
            }
        }

        hasObj = tmpHasObj
        clsOffset = tmpClsOffset
        numClasses = tmpNumClasses
        featuresPerDet = clsOffset + numClasses

        classNames = when {
            numClasses == cocoLabels.size -> cocoLabels
            numClasses == 1 -> listOf("bottle") // your fine-tuned nc=1 models
            numClasses > 1 -> List(numClasses) { i -> "cls$i" }
            else -> emptyList()
        }

        Log.d(
            TAG,
            "Parsed: numDet=$numDetections numClasses=$numClasses channelsLast=$channelsLast hasObj=$hasObj clsOffset=$clsOffset featuresPerDet=$featuresPerDet"
        )

        val expectedInputFloats = inputWidth * inputHeight * 3
        inputByteBuffer = ByteBuffer
            .allocateDirect(4 * expectedInputFloats)
            .order(ByteOrder.nativeOrder())

        if (channelsLast) {
            outChannelsLast = Array(1) { Array(numDetections) { FloatArray(featuresPerDet) } }
            outChannelsFirst = null
        } else {
            outChannelsFirst = Array(1) { Array(featuresPerDet) { FloatArray(numDetections) } }
            outChannelsLast = null
        }
    }

    private fun createInterpreter(assetName: String): Interpreter {
        val model = FileUtil.loadMappedFile(context, assetName)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseXNNPACK(false)
        }
        Log.d(TAG, "Creating interpreter for $assetName (useXNNPACK=false)")
        return Interpreter(model, options)
    }

    /**
     * targetLabelForDiagnostics:
     * - if non-null, compute max score for that label across ALL candidates, even if below threshold.
     * - Works for both COCO-80 and single-class (mapped to "bottle").
     *
     * Also computes maxAnyScore + maxAnyLabel (best over all classes/candidates).
     */
    fun detectTimed(
        input: FloatArray,
        scoreThreshold: Float = 0.35f,
        iouThreshold: Float = 0.45f,
        maxDetections: Int = 100,
        targetLabelForDiagnostics: String? = null
    ): TimedDetections {
        val expected = inputWidth * inputHeight * 3
        require(input.size == expected) {
            "Bad input length ${input.size}, expected $expected"
        }

        if (inputType != DataType.FLOAT32) {
            Log.w(TAG, "Input tensor type is $inputType for $modelAssetName; current code packs FLOAT32.")
        }

        val targetClassIdx: Int = if (targetLabelForDiagnostics == null) {
            -1
        } else {
            classNames.indexOf(targetLabelForDiagnostics)
        }

        inputByteBuffer.rewind()
        inputByteBuffer.asFloatBuffer().apply {
            rewind()
            put(input)
        }

        val candidates = ArrayList<Detection>(64)
        var maxTargetScore = 0.0

        var maxAnyScore = 0.0
        var maxAnyClassIdx = -1

        val inferStartNs = SystemClock.elapsedRealtimeNanos()

        try {
            if (channelsLast) {
                val out = outChannelsLast ?: return TimedDetections(emptyList(), 0.0, 0.0, 0.0, "")
                synchronized(interpreterLock) {
                    interpreter.run(inputByteBuffer, out)
                }

                val inferEndNs = SystemClock.elapsedRealtimeNanos()
                val inferenceMs = (inferEndNs - inferStartNs) / 1_000_000.0

                for (i in 0 until numDetections) {
                    val row = out[0][i]
                    val cx = row[0]
                    val cy = row[1]
                    val w = row[2]
                    val h = row[3]

                    if (numClasses > 0) {
                        val obj = if (hasObj) row[4] else 1f

                        // Max-any-class for this candidate
                        var bestIdx = 0
                        var bestCls = row[clsOffset]
                        for (c in 1 until numClasses) {
                            val s = row[clsOffset + c]
                            if (s > bestCls) { bestCls = s; bestIdx = c }
                        }
                        val bestScore = (obj * bestCls).toDouble()
                        if (bestScore > maxAnyScore) {
                            maxAnyScore = bestScore
                            maxAnyClassIdx = bestIdx
                        }

                        // Target diagnostics
                        if (targetClassIdx in 0 until numClasses) {
                            val clsScore = row[clsOffset + targetClassIdx]
                            val tScore = (obj * clsScore).toDouble()
                            if (tScore > maxTargetScore) maxTargetScore = tScore
                        }

                        // Normal detection (thresholded)
                        val score = (obj * bestCls)
                        if (score >= scoreThreshold) {
                            val left = cx - w / 2f
                            val top = cy - h / 2f
                            val right = cx + w / 2f
                            val bottom = cy + h / 2f
                            candidates += Detection(RectF(left, top, right, bottom), classNames[bestIdx], score)
                        }
                    }
                }

                val finalDetections = nonMaxSuppression(candidates, iouThreshold, maxDetections)
                val maxAnyLabel = if (maxAnyClassIdx in 0 until classNames.size) classNames[maxAnyClassIdx] else ""
                return TimedDetections(finalDetections, inferenceMs, maxTargetScore, maxAnyScore, maxAnyLabel)
            } else {
                val out = outChannelsFirst ?: return TimedDetections(emptyList(), 0.0, 0.0, 0.0, "")
                synchronized(interpreterLock) {
                    interpreter.run(inputByteBuffer, out)
                }

                val inferEndNs = SystemClock.elapsedRealtimeNanos()
                val inferenceMs = (inferEndNs - inferStartNs) / 1_000_000.0

                for (i in 0 until numDetections) {
                    val cx = out[0][0][i]
                    val cy = out[0][1][i]
                    val w = out[0][2][i]
                    val h = out[0][3][i]

                    if (numClasses > 0) {
                        val obj = if (hasObj) out[0][4][i] else 1f

                        // Max-any-class for this candidate
                        var bestIdx = 0
                        var bestCls = out[0][clsOffset][i]
                        for (c in 1 until numClasses) {
                            val s = out[0][clsOffset + c][i]
                            if (s > bestCls) { bestCls = s; bestIdx = c }
                        }
                        val bestScore = (obj * bestCls).toDouble()
                        if (bestScore > maxAnyScore) {
                            maxAnyScore = bestScore
                            maxAnyClassIdx = bestIdx
                        }

                        // Target diagnostics
                        if (targetClassIdx in 0 until numClasses) {
                            val clsScore = out[0][clsOffset + targetClassIdx][i]
                            val tScore = (obj * clsScore).toDouble()
                            if (tScore > maxTargetScore) maxTargetScore = tScore
                        }

                        // Normal detection (thresholded)
                        val score = (obj * bestCls)
                        if (score >= scoreThreshold) {
                            val left = cx - w / 2f
                            val top = cy - h / 2f
                            val right = cx + w / 2f
                            val bottom = cy + h / 2f
                            candidates += Detection(RectF(left, top, right, bottom), classNames[bestIdx], score)
                        }
                    }
                }

                val finalDetections = nonMaxSuppression(candidates, iouThreshold, maxDetections)
                val maxAnyLabel = if (maxAnyClassIdx in 0 until classNames.size) classNames[maxAnyClassIdx] else ""
                return TimedDetections(finalDetections, inferenceMs, maxTargetScore, maxAnyScore, maxAnyLabel)
            }
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "Interpreter run failed for $modelAssetName: ${e.message}", e)
            val maxAnyLabel = if (maxAnyClassIdx in 0 until classNames.size) classNames[maxAnyClassIdx] else ""
            return TimedDetections(emptyList(), 0.0, maxTargetScore, maxAnyScore, maxAnyLabel)
        } catch (t: Throwable) {
            Log.e(TAG, "Unexpected failure in detectTimed for $modelAssetName: ${t.message}", t)
            val maxAnyLabel = if (maxAnyClassIdx in 0 until classNames.size) classNames[maxAnyClassIdx] else ""
            return TimedDetections(emptyList(), 0.0, maxTargetScore, maxAnyScore, maxAnyLabel)
        }
    }

    private fun nonMaxSuppression(
        detections: List<Detection>,
        iouThreshold: Float,
        maxDetections: Int
    ): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        val sorted = detections.sortedByDescending { it.score }
        val selected = mutableListOf<Detection>()

        for (det in sorted) {
            var ok = true
            for (sel in selected) {
                if (sel.label == det.label && iou(sel.box, det.box) > iouThreshold) {
                    ok = false
                    break
                }
            }
            if (ok) {
                selected += det
                if (selected.size >= maxDetections) break
            }
        }
        return selected
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        if (interRight <= interLeft || interBottom <= interTop) return 0f

        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val unionArea = a.width() * a.height() + b.width() * b.height() - interArea
        if (unionArea <= 0f) return 0f
        return interArea / unionArea
    }

    fun close() {
        synchronized(interpreterLock) {
            interpreter.close()
        }
    }
}