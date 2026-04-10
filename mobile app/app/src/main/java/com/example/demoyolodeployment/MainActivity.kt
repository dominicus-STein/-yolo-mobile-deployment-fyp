package com.example.demoyolodeployment

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.view.Surface
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {

    private lateinit var overlay: OverlayView
    private lateinit var previewView: PreviewView

    private lateinit var spinnerModel: Spinner
    private lateinit var btnToggleDetection: Button
    private lateinit var btnBenchmark: Button

    @Volatile private var detector: YoloDetector? = null

    private lateinit var converter: RgbaToBitmapConverter
    private lateinit var analysisExecutor: ExecutorService

    private var cameraProvider: ProcessCameraProvider? = null
    private var previewUseCase: Preview? = null
    private var analysisUseCase: ImageAnalysis? = null

    private var rgbaBitmap: Bitmap? = null
    private var resizedBitmap: Bitmap? = null
    private var pixelBuffer: IntArray? = null
    private var inputBuffer: FloatArray? = null

    private lateinit var benchmarkRunner: BenchmarkRunner

    private val isSwappingModel = AtomicBoolean(false)

    // Detection is OFF by default
    private val detectionEnabled = AtomicBoolean(false)

    // Benchmark temporarily forces inference even if detection toggle is OFF
    private val benchmarkForcingDetection = AtomicBoolean(false)

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
    }

    private data class ModelVariant(val displayName: String, val assetName: String)

    private val modelVariants: List<ModelVariant> = listOf(
        ModelVariant("base_silu", "yolo11n_base_silu_fp16_640.tflite"),
        ModelVariant("act_relu", "yolo11n_act_relu_fp16_640.tflite"),
        ModelVariant("act_hardswish", "yolo11n_act_hardswish_fp16_640.tflite"),
        ModelVariant("psa_noattn", "yolo11n_psa_noattn_fp16_640.tflite"),
        ModelVariant("psa_noffn", "yolo11n_psa_noffn_fp16_640.tflite")
    )

    @Volatile private var currentModelVariant: ModelVariant = modelVariants.first()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)

        spinnerModel = findViewById(R.id.spinnerModel)
        btnToggleDetection = findViewById(R.id.btnToggleDetection)
        btnBenchmark = findViewById(R.id.btnBenchmark)

        converter = RgbaToBitmapConverter()
        analysisExecutor = Executors.newSingleThreadExecutor()

        // Load default model (baseline). Detection stays OFF.
        currentModelVariant = modelVariants.first()
        detector = YoloDetector(this, modelAssetName = currentModelVariant.assetName)

        overlay.visibility = View.VISIBLE
        overlay.setDetections(emptyList(), 1f, 1f)
        setDetectionUiState(enabled = false)

        setupModelSpinner()
        setupButtons()

        benchmarkRunner = BenchmarkRunner(
            context = this,
            onShowMessage = { msg ->
                runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
            },
            onModelSwitch = { spec ->
                switchModelSafe(spec)
            },
            onStopped = {
                runOnUiThread {
                    benchmarkForcingDetection.set(false)
                    detectionEnabled.set(false)
                    setDetectionUiState(false)
                    overlay.setDetections(emptyList(), 1f, 1f)
                    btnBenchmark.text = "Benchmark"
                }
            }
        )

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermission.launch(Manifest.permission.CAMERA)
        } else {
            startCamera()
        }
    }

    override fun onResume() {
        super.onResume()
        val rot = previewView.display?.rotation ?: Surface.ROTATION_0
        previewUseCase?.targetRotation = rot
        analysisUseCase?.targetRotation = rot
    }

    private fun setupModelSpinner() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            modelVariants.map { it.displayName }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerModel.adapter = adapter

        spinnerModel.setSelection(0, false)

        spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val chosen = modelVariants.getOrNull(position) ?: return
                if (chosen.assetName == currentModelVariant.assetName) return

                if (benchmarkRunner.isRunning()) {
                    Toast.makeText(
                        this@MainActivity,
                        "Stop benchmark before switching models",
                        Toast.LENGTH_SHORT
                    ).show()
                    val idx = modelVariants.indexOfFirst { it.assetName == currentModelVariant.assetName }
                    if (idx >= 0) spinnerModel.setSelection(idx, false)
                    return
                }

                val ok = switchModelSafe(
                    BenchmarkRunner.ModelSpec(
                        id = chosen.displayName,
                        assetName = chosen.assetName
                    )
                )

                if (ok) {
                    currentModelVariant = chosen
                    overlay.setDetections(emptyList(), 1f, 1f)
                    Toast.makeText(
                        this@MainActivity,
                        "Model: ${chosen.displayName}",
                        Toast.LENGTH_SHORT
                    ).show()
                } else {
                    val idx = modelVariants.indexOfFirst { it.assetName == currentModelVariant.assetName }
                    if (idx >= 0) spinnerModel.setSelection(idx, false)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setupButtons() {
        btnToggleDetection.setOnClickListener {
            if (benchmarkRunner.isRunning()) {
                Toast.makeText(
                    this,
                    "Benchmark running: detection is controlled by benchmark",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            val newState = !detectionEnabled.get()
            detectionEnabled.set(newState)
            setDetectionUiState(newState)

            if (!newState) {
                overlay.setDetections(emptyList(), 1f, 1f)
            }
        }

        btnBenchmark.setOnClickListener {
            if (benchmarkRunner.isRunning()) {
                benchmarkRunner.stop()
                return@setOnClickListener
            }

            val thermal = getCurrentThermalStatus()
            if (thermal != 0) {
                Toast.makeText(
                    this,
                    "Benchmark blocked: thermal=$thermal (must be 0 to start)",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            benchmarkRunner.setCurrentModel(
                modelId = currentModelVariant.displayName,
                assetName = currentModelVariant.assetName
            )

            // Force inference ON during benchmark, but keep detection toggle OFF by default
            benchmarkForcingDetection.set(true)
            detectionEnabled.set(false)
            setDetectionUiState(false)
            overlay.setDetections(emptyList(), 1f, 1f)

            val ok = benchmarkRunner.start()
            if (!ok) {
                benchmarkForcingDetection.set(false)
                Toast.makeText(this, "Benchmark: FAILED (see toast/log)", Toast.LENGTH_SHORT).show()
            } else {
                btnBenchmark.text = "Stop"
                Toast.makeText(this, "Benchmark: START", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setDetectionUiState(enabled: Boolean) {
        btnToggleDetection.text = if (enabled) "Detection: ON" else "Detection: OFF"
    }

    private fun getCurrentThermalStatus(): Int {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        return if (android.os.Build.VERSION.SDK_INT >= 29) {
            pm.currentThermalStatus
        } else {
            0
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)

        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider
            provider.unbindAll()

            val rot = previewView.display?.rotation ?: Surface.ROTATION_0

            previewUseCase = Preview.Builder()
                .setTargetRotation(rot)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            analysisUseCase = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetRotation(rot)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                previewUseCase,
                analysisUseCase
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val det = detector
        if (det == null || isSwappingModel.get()) {
            imageProxy.close()
            return
        }

        val shouldRunInference = detectionEnabled.get() || benchmarkForcingDetection.get()

        val frameStartNs = SystemClock.elapsedRealtimeNanos()

        try {
            val srcW = imageProxy.width
            val srcH = imageProxy.height

            val rgba = rgbaBitmap?.takeIf { it.width == srcW && it.height == srcH }
                ?: Bitmap.createBitmap(srcW, srcH, Bitmap.Config.ARGB_8888).also { rgbaBitmap = it }

            converter.rgbaToBitmap(imageProxy, rgba)

            if (!shouldRunInference) {
                return
            }

            val iw = det.inputWidth
            val ih = det.inputHeight

            val resized = resizedBitmap?.takeIf { it.width == iw && it.height == ih }
                ?: Bitmap.createBitmap(iw, ih, Bitmap.Config.ARGB_8888).also { resizedBitmap = it }

            val canvas = android.graphics.Canvas(resized)
            canvas.drawBitmap(
                rgba,
                android.graphics.Rect(0, 0, rgba.width, rgba.height),
                android.graphics.Rect(0, 0, iw, ih),
                null
            )

            val pixels = pixelBuffer?.takeIf { it.size == iw * ih }
                ?: IntArray(iw * ih).also { pixelBuffer = it }
            resized.getPixels(pixels, 0, iw, 0, 0, iw, ih)

            val input = inputBuffer?.takeIf { it.size == iw * ih * 3 }
                ?: FloatArray(iw * ih * 3).also { inputBuffer = it }

            if (!det.isNCHW) {
                var idx = 0
                for (p in pixels) {
                    input[idx++] = ((p shr 16) and 0xFF) / 255f
                    input[idx++] = ((p shr 8) and 0xFF) / 255f
                    input[idx++] = (p and 0xFF) / 255f
                }
            } else {
                val plane = iw * ih
                var i = 0
                while (i < pixels.size) {
                    val p = pixels[i]
                    input[i] = ((p shr 16) and 0xFF) / 255f
                    input[plane + i] = ((p shr 8) and 0xFF) / 255f
                    input[2 * plane + i] = (p and 0xFF) / 255f
                    i++
                }
            }

            // request diagnostics for bottle confidence even when below threshold
            val timed = det.detectTimed(
                input = input,
                scoreThreshold = 0.35f,
                iouThreshold = 0.45f,
                maxDetections = 100,
                targetLabelForDiagnostics = "bottle"
            )

            val detections = timed.detections
            val inferenceMs = timed.inferenceMs
            val maxBottleScoreAny = timed.maxTargetScore

            val frameEndNs = SystemClock.elapsedRealtimeNanos()
            val totalMs = (frameEndNs - frameStartNs) / 1_000_000.0

            benchmarkRunner.onFrame(
                frameTsNs = frameEndNs,
                totalLatencyMs = totalMs,
                inferenceMs = inferenceMs,
                detections = timed.detections,
                maxTargetScoreThisFrame = timed.maxTargetScore,
                maxAnyScoreThisFrame = timed.maxAnyScore,
                maxAnyLabelThisFrame = timed.maxAnyLabel
            )

            val vw = previewView.width
            val vh = previewView.height
            if (vw != 0 && vh != 0) {
                val sx = vw.toFloat() / det.inputWidth
                val sy = vh.toFloat() / det.inputHeight
                runOnUiThread {
                    overlay.setDetections(detections, sx, sy)
                }
            }
        } finally {
            imageProxy.close()
        }
    }

    private fun switchModelSafe(spec: BenchmarkRunner.ModelSpec): Boolean {
        isSwappingModel.set(true)
        return try {
            detector?.close()
            detector = YoloDetector(this, modelAssetName = spec.assetName)
            true
        } catch (t: Throwable) {
            runOnUiThread {
                Toast.makeText(
                    this,
                    "Model load failed: ${spec.assetName}\n${t.javaClass.simpleName}: ${t.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
            try {
                detector = YoloDetector(this, modelAssetName = modelVariants.first().assetName)
                currentModelVariant = modelVariants.first()
                spinnerModel.setSelection(0, false)
            } catch (_: Throwable) {
                detector = null
            }
            false
        } finally {
            isSwappingModel.set(false)
        }
    }

    override fun onDestroy() {
        analysisUseCase?.clearAnalyzer()
        cameraProvider?.unbindAll()

        analysisExecutor.shutdown()
        detector?.close()

        super.onDestroy()
    }
}