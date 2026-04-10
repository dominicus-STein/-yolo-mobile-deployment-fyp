package com.example.demoyolodeployment

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import java.io.File
import java.io.FileWriter
import java.util.Locale
import kotlin.math.sqrt

class BenchmarkRunner(
    private val context: Context,
    private val onShowMessage: (String) -> Unit,
    private val onModelSwitch: (ModelSpec) -> Boolean,
    private val onStopped: () -> Unit
) {
    data class ModelSpec(val id: String, val assetName: String)

    private val conditions = listOf(
        "COUNT_1", "COUNT_2", "COUNT_3",
        "DIST_NEAR", "DIST_MID", "DIST_FAR"
    )

    private val warmupMs = 5_000L
    private val measureMs = 30_000L
    private val absenceMs = 30_000L
    private val absenceGraceMs = 3_000L

    private val trials = 10
    private val trialWindowMs = 1_000L
    private val trialsSpanMs = trials * trialWindowMs

    private val targetLabel = "bottle"

    private enum class Phase { OFF, WARMUP, MEASURE, ABSENCE }

    private val main = Handler(Looper.getMainLooper())
    private var phase: Phase = Phase.OFF

    private var conditionIndex = 0
    private var phaseStartMs = 0L

    private var writer: FileWriter? = null
    private var csvFile: File? = null

    private val agg = Aggregator()

    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private var samplerRunning = false

    @Volatile
    private var currentModel: ModelSpec = ModelSpec(
        id = "base_silu",
        assetName = "yolo11n_base_silu_fp16_640.tflite"
    )

    fun isRunning(): Boolean = phase != Phase.OFF

    fun setCurrentModel(modelId: String, assetName: String) {
        currentModel = ModelSpec(modelId, assetName)
    }

    fun start(): Boolean {
        if (isRunning()) return true

        val thermal = getCurrentThermalStatus()
        if (thermal != 0) {
            onShowMessage("Benchmark blocked: thermal=$thermal (must be 0 to start)")
            return false
        }

        if (!assetExists(currentModel.assetName)) {
            onShowMessage("Missing model asset: ${currentModel.assetName}")
            return false
        }
        if (!assetExists("labels.txt")) {
            onShowMessage("Missing asset: labels.txt")
            return false
        }

        conditionIndex = 0

        val dir = File(context.getExternalFilesDir(null), "benchmarks")
        dir.mkdirs()
        val ts = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
            .format(java.util.Date())
        csvFile = File(dir, "yolo_benchmark_${currentModel.id}_$ts.csv")
        writer = FileWriter(csvFile!!, false)

        writeCsvHeader()
        onShowMessage("Benchmark file: ${csvFile!!.name}")

        runNextCondition()
        return true
    }

    fun stop() {
        if (!isRunning()) return

        endSampling()
        phase = Phase.OFF
        main.removeCallbacksAndMessages(null)

        writer?.flush()
        writer?.close()
        writer = null

        onShowMessage("Benchmark stopped. CSV saved.")
        onStopped()
    }

    /**
     * maxTargetScoreThisFrame: best "bottle" score in this frame (even below threshold)
     * maxAnyScoreThisFrame: best score over ANY class in this frame (even below threshold)
     * maxAnyLabelThisFrame: label associated with maxAnyScoreThisFrame (if known)
     */
    fun onFrame(
        frameTsNs: Long,
        totalLatencyMs: Double,
        inferenceMs: Double,
        detections: List<Detection>,
        maxTargetScoreThisFrame: Double,
        maxAnyScoreThisFrame: Double,
        maxAnyLabelThisFrame: String
    ) {
        val p = phase
        if (p == Phase.OFF) return

        agg.onFrameTs(frameTsNs)

        if (p == Phase.MEASURE) {
            agg.onMeasureFrame(
                nowMs = SystemClock.elapsedRealtime(),
                phaseStartMs = phaseStartMs,
                totalLatencyMs = totalLatencyMs,
                inferenceMs = inferenceMs,
                detections = detections,
                targetLabel = targetLabel,
                trialsSpanMs = trialsSpanMs,
                trials = trials,
                trialWindowMs = trialWindowMs,
                maxTargetScoreThisFrame = maxTargetScoreThisFrame,
                maxAnyScoreThisFrame = maxAnyScoreThisFrame,
                maxAnyLabelThisFrame = maxAnyLabelThisFrame
            )
        } else if (p == Phase.ABSENCE) {
            agg.onAbsenceFrame(
                nowMs = SystemClock.elapsedRealtime(),
                detections = detections,
                targetLabel = targetLabel,
                graceMs = absenceGraceMs
            )
        }
    }

    private fun runNextCondition() {
        if (conditionIndex >= conditions.size) {
            stop()
            onShowMessage("Benchmark complete. CSV saved.")
            return
        }

        val model = currentModel
        val condition = conditions[conditionIndex]

        val switched = onModelSwitch(model)
        if (!switched) {
            onShowMessage("Stopping benchmark (model load failed).")
            stop()
            return
        }

        agg.reset()
        agg.modelId = model.id
        agg.conditionId = condition

        phase = Phase.WARMUP
        phaseStartMs = SystemClock.elapsedRealtime()
        onShowMessage("SETUP: ${model.id} × $condition (Warm-up 5s)")
        beginSampling()

        main.postDelayed({ startMeasurePhase() }, warmupMs)
    }

    private fun startMeasurePhase() {
        if (phase != Phase.WARMUP) return

        phase = Phase.MEASURE
        phaseStartMs = SystemClock.elapsedRealtime()
        agg.markMeasureStart(phaseStartMs)

        onShowMessage("MEASURE: ${agg.modelId} × ${agg.conditionId} (30s)")
        main.postDelayed({ startAbsencePhase() }, measureMs)
    }

    private fun startAbsencePhase() {
        if (phase != Phase.MEASURE) return

        phase = Phase.ABSENCE
        phaseStartMs = SystemClock.elapsedRealtime()
        agg.markAbsenceStart(phaseStartMs)

        onShowMessage("ABSENCE: remove bottle(s) (30s)")
        main.postDelayed({ finishCondition() }, absenceMs)
    }

    private fun finishCondition() {
        if (phase != Phase.ABSENCE) return

        endSampling()

        val row = agg.buildCsvRow(trials = trials)
        appendCsvRow(row)

        onShowMessage("Saved: ${agg.modelId} × ${agg.conditionId}")

        conditionIndex++
        runNextCondition()
    }

    private fun assetExists(assetName: String): Boolean {
        return try {
            context.assets.open(assetName).close()
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun getCurrentThermalStatus(): Int {
        return if (Build.VERSION.SDK_INT >= 29) {
            powerManager.currentThermalStatus
        } else {
            0
        }
    }

    private fun beginSampling() {
        if (samplerRunning) return
        samplerRunning = true

        val r = object : Runnable {
            override fun run() {
                if (!samplerRunning) return

                if (Build.VERSION.SDK_INT >= 29) {
                    val t = powerManager.currentThermalStatus
                    agg.thermalMax = maxOf(agg.thermalMax, t)
                }

                val mi = Debug.MemoryInfo()
                Debug.getMemoryInfo(mi)
                val pssKb = mi.totalPss
                agg.peakPssKb = maxOf(agg.peakPssKb, pssKb)

                main.postDelayed(this, 1000L)
            }
        }
        main.post(r)
    }

    private fun endSampling() {
        samplerRunning = false
    }

    private fun writeCsvHeader() {
        val header = listOf(
            "timestamp",
            "model_id",
            "condition",
            "fps_avg",
            "fps_min",
            "fps_std",
            "latency_avg_ms",
            "latency_p90_ms",
            "inference_avg_ms",
            "inference_p90_ms",
            "inference_p99_ms",
            "detection_rate_trials",
            "avg_conf_success",
            "false_pos",
            "thermal_status_max",
            "peak_pss_mb",
            "avg_conf_best_any",
            "max_conf_best_any",
            "trial_best_conf_any",
            "avg_any_conf_best_any",
            "max_any_conf_best_any",
            "trial_best_any_conf_any",
            "trial_best_any_label_any"
        ).joinToString(",") + "\n"
        writer!!.write(header)
        writer!!.flush()
    }

    private fun appendCsvRow(row: String) {
        writer?.write(row)
        writer?.flush()
    }

    private class Aggregator {
        var modelId: String = ""
        var conditionId: String = ""

        private var lastFrameTsNs: Long = 0L
        private val fpsSamples = ArrayList<Double>(4096)

        private val totalLatencySamples = ArrayList<Double>(4096)
        private val inferenceSamples = ArrayList<Double>(4096)

        // Success tracking (above threshold, because detections list already thresholded)
        private val trialHit = BooleanArray(10) { false }
        private val trialBestConfSuccess = DoubleArray(10) { 0.0 }

        // Best bottle confidence per trial regardless of threshold
        private val trialBestConfAny = DoubleArray(10) { 0.0 }

        // Best any-class confidence per trial regardless of threshold
        private val trialBestAnyConfAny = DoubleArray(10) { 0.0 }
        private val trialBestAnyLabelAny = Array(10) { "" }

        // Counts at most 1 false positive per frame, by design
        private var falsePosCount = 0
        private var absenceStartMs: Long = 0L

        var thermalMax: Int = 0
        var peakPssKb: Int = 0

        fun reset() {
            lastFrameTsNs = 0L
            fpsSamples.clear()
            totalLatencySamples.clear()
            inferenceSamples.clear()

            for (i in trialHit.indices) {
                trialHit[i] = false
                trialBestConfSuccess[i] = 0.0
                trialBestConfAny[i] = 0.0
                trialBestAnyConfAny[i] = 0.0
                trialBestAnyLabelAny[i] = ""
            }

            falsePosCount = 0
            absenceStartMs = 0L
            thermalMax = 0
            peakPssKb = 0
        }

        fun onFrameTs(tsNs: Long) {
            if (lastFrameTsNs != 0L) {
                val dtNs = tsNs - lastFrameTsNs
                if (dtNs > 0) fpsSamples.add(1e9 / dtNs.toDouble())
            }
            lastFrameTsNs = tsNs
        }

        fun markMeasureStart(ms: Long) {
            // no-op
        }

        fun markAbsenceStart(ms: Long) {
            absenceStartMs = ms
        }

        fun onMeasureFrame(
            nowMs: Long,
            phaseStartMs: Long,
            totalLatencyMs: Double,
            inferenceMs: Double,
            detections: List<Detection>,
            targetLabel: String,
            trialsSpanMs: Long,
            trials: Int,
            trialWindowMs: Long,
            maxTargetScoreThisFrame: Double,
            maxAnyScoreThisFrame: Double,
            maxAnyLabelThisFrame: String
        ) {
            totalLatencySamples.add(totalLatencyMs)
            inferenceSamples.add(inferenceMs)

            val elapsed = nowMs - phaseStartMs
            if (elapsed in 0 until trialsSpanMs) {
                val trialIdx = (elapsed / trialWindowMs).toInt()
                if (trialIdx in 0 until trials) {
                    // Bottle best-any
                    if (maxTargetScoreThisFrame > trialBestConfAny[trialIdx]) {
                        trialBestConfAny[trialIdx] = maxTargetScoreThisFrame
                    }

                    // Any-class best-any
                    if (maxAnyScoreThisFrame > trialBestAnyConfAny[trialIdx]) {
                        trialBestAnyConfAny[trialIdx] = maxAnyScoreThisFrame
                        trialBestAnyLabelAny[trialIdx] = maxAnyLabelThisFrame
                    }

                    // Success/confidence only if above threshold (detections list)
                    var bestSuccess = 0.0
                    var hit = false
                    for (d in detections) {
                        if (d.label == targetLabel) {
                            hit = true
                            if (d.score.toDouble() > bestSuccess) bestSuccess = d.score.toDouble()
                        }
                    }
                    if (hit) {
                        trialHit[trialIdx] = true
                        if (bestSuccess > trialBestConfSuccess[trialIdx]) {
                            trialBestConfSuccess[trialIdx] = bestSuccess
                        }
                    }
                }
            }
        }

        fun onAbsenceFrame(
            nowMs: Long,
            detections: List<Detection>,
            targetLabel: String,
            graceMs: Long
        ) {
            if (absenceStartMs != 0L && nowMs - absenceStartMs < graceMs) return

            for (d in detections) {
                if (d.label == targetLabel) {
                    falsePosCount++
                    break
                }
            }
        }

        fun buildCsvRow(trials: Int): String {
            val now = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                .format(java.util.Date())

            val fpsAvg = mean(fpsSamples)
            val fpsMin = fpsSamples.minOrNull() ?: 0.0
            val fpsStd = stddev(fpsSamples)

            val latAvg = mean(totalLatencySamples)
            val latP90 = percentile(totalLatencySamples, 90.0)

            val infAvg = mean(inferenceSamples)
            val infP90 = percentile(inferenceSamples, 90.0)
            val infP99 = percentile(inferenceSamples, 99.0)

            val hitCount = trialHit.count { it }
            val detRate = "'$hitCount/$trials" // Excel-safe

            val confsSuccess = ArrayList<Double>()
            for (i in 0 until trials) {
                if (trialHit[i]) confsSuccess.add(trialBestConfSuccess[i])
            }
            val confAvgSuccess = if (confsSuccess.isEmpty()) 0.0 else mean(confsSuccess)

            val peakPssMb = peakPssKb / 1024.0

            val avgBestBottleAny = mean(trialBestConfAny.toList())
            val maxBestBottleAny = trialBestConfAny.maxOrNull() ?: 0.0
            val trialBestBottleAnyStr = trialBestConfAny.joinToString(";") { fmt(it) }

            val avgBestAny = mean(trialBestAnyConfAny.toList())
            val maxBestAny = trialBestAnyConfAny.maxOrNull() ?: 0.0
            val trialBestAnyStr = trialBestAnyConfAny.joinToString(";") { fmt(it) }
            val trialBestAnyLabelStr =
                trialBestAnyLabelAny.joinToString(";") { sanitizeLabel(it) }

            return listOf(
                now,
                modelId,
                conditionId,
                fmt(fpsAvg),
                fmt(fpsMin),
                fmt(fpsStd),
                fmt(latAvg),
                fmt(latP90),
                fmt(infAvg),
                fmt(infP90),
                fmt(infP99),
                detRate,
                fmt(confAvgSuccess),
                falsePosCount.toString(),
                thermalMax.toString(),
                fmt(peakPssMb),
                fmt(avgBestBottleAny),
                fmt(maxBestBottleAny),
                trialBestBottleAnyStr,
                fmt(avgBestAny),
                fmt(maxBestAny),
                trialBestAnyStr,
                trialBestAnyLabelStr
            ).joinToString(",") + "\n"
        }

        private fun sanitizeLabel(s: String): String {
            return s.replace(",", "_")
        }

        private fun fmt(x: Double): String = String.format(Locale.US, "%.3f", x)

        private fun mean(x: List<Double>): Double {
            if (x.isEmpty()) return 0.0
            var s = 0.0
            for (v in x) s += v
            return s / x.size
        }

        private fun stddev(x: List<Double>): Double {
            if (x.size < 2) return 0.0
            val m = mean(x)
            var ss = 0.0
            for (v in x) {
                val d = v - m
                ss += d * d
            }
            return sqrt(ss / (x.size - 1))
        }

        private fun percentile(x: List<Double>, p: Double): Double {
            if (x.isEmpty()) return 0.0
            val sorted = x.sorted()
            val rank = (p / 100.0) * (sorted.size - 1)
            val lo = rank.toInt()
            val hi = kotlin.math.min(lo + 1, sorted.size - 1)
            val w = rank - lo
            return sorted[lo] * (1.0 - w) + sorted[hi] * w
        }
    }
}