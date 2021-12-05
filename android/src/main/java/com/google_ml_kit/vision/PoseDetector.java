package com.google_ml_kit.vision;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseLandmark;
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;
import com.google_ml_kit.ApiDetectorInterface;
import com.google_ml_kit.vision.classification.PoseClassifierProcessor;
import com.google_ml_kit.vision.classification.RepetitionCounter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;

// Detector to the pose landmarks present in a image.
// Creates an abstraction over PoseDetector provided by ml kit.
public class PoseDetector implements ApiDetectorInterface {
    private static final String START_WITHOUT_CLASSIFIER = "vision#startPoseDetectorWithoutCl";
    private static final String START_WITH_CLASSIFIER = "vision#startPoseDetectorWithCl";
    private static final String START_ACTIVITY = "vision#startPoseDetectorActivity";
    private static final String CLOSE = "vision#closePoseDetector";

    private final Context context;
    private com.google.mlkit.vision.pose.PoseDetector poseDetector;
    private PoseClassifierProcessor poseClassifierProcessor;
    private final Executor classificationExecutor;

    public PoseDetector(Context context) {
        this.context = context;
        classificationExecutor = Executors.newSingleThreadExecutor();
    }

    protected static class PoseWithClassification {
        private final Pose pose;
        private final List<String> classificationResult;

        public PoseWithClassification(Pose pose, List<String> classificationResult) {
            this.pose = pose;
            this.classificationResult = classificationResult;
        }


        public Pose getPose() {
            return pose;
        }

        public List<String> getClassificationResult() {
            return classificationResult;
        }
    }

    @Override
    public List<String> getMethodsKeys() {
        return new ArrayList<>(
                Arrays.asList(START_WITHOUT_CLASSIFIER, START_WITH_CLASSIFIER, START_ACTIVITY, CLOSE)
        );
    }

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull MethodChannel.Result result) {
        String method = call.method;
        if (method.equals(START_WITHOUT_CLASSIFIER) || method.equals(START_WITH_CLASSIFIER) || method.equals(START_ACTIVITY)) {
            handleDetection(call, result);
        } else if (method.equals(CLOSE)) {
            closeDetector();
            result.success(null);
        } else {
            result.notImplemented();
        }
    }

    private void handleDetection(MethodCall call, final MethodChannel.Result result) {
        String methodName = call.method;
        Map<String, Object> imageData = (Map<String, Object>) call.argument("imageData");
        InputImage inputImage = InputImageConverter.getInputImageFromData(imageData, context, result);
        if (inputImage == null) return;

        Map<String, Object> options = call.argument("options");
        if (options == null) {
            result.error("PoseDetectorError", "Invalid options", null);
            return;
        }

        String model = (String) options.get("type");
        String mode = (String) options.get("mode");
        int detectorMode = AccuratePoseDetectorOptions.STREAM_MODE;

        // Base pose detector with streaming frames, when depending on the pose-detection sdk
//        PoseDetectorOptions detectorOptions = new PoseDetectorOptions.Builder()
//                .setDetectorMode(detectorMode)
//                .build();
//        poseDetector = PoseDetection.getClient(detectorOptions);

        // Accurate pose detector with streaming frames, when depending on the pose-detection-accurate sdk
        AccuratePoseDetectorOptions detectorOptions = new AccuratePoseDetectorOptions.Builder()
                .setDetectorMode(detectorMode)
                .build();
        poseDetector = PoseDetection.getClient(detectorOptions);

        if (methodName.equals(START_WITHOUT_CLASSIFIER)) {
            poseDetector.process(inputImage)
                    .addOnSuccessListener(
                            (OnSuccessListener<Pose>) pose -> {
                                List<Map<String, Object>> poseList = new ArrayList<>();

                                if (!pose.getAllPoseLandmarks().isEmpty()) {
                                    Map<String, Object> poseMap = new HashMap<String, Object>();
                                    List<Map<String, Object>> landmarks = new ArrayList<>();
                                    for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
                                        Map<String, Object> landmarkMap = new HashMap<>();
                                        landmarkMap.put("type", poseLandmark.getLandmarkType());
                                        landmarkMap.put("x", poseLandmark.getPosition3D().getX());
                                        landmarkMap.put("y", poseLandmark.getPosition3D().getY());
                                        landmarkMap.put("z", poseLandmark.getPosition3D().getZ());
                                        landmarkMap.put("likelihood", poseLandmark.getInFrameLikelihood());
                                        landmarks.add(landmarkMap);
                                    }
                                    poseMap.put("landmarks", landmarks);
                                    poseMap.put("name", PoseDataStorage.getPose());
                                    poseMap.put("accuracy", PoseDataStorage.getAccuracy());
                                    poseList.add(poseMap);
                                }
                                result.success(poseList);
                            })
                    .addOnFailureListener(e -> result.error("PoseDetectorError", e.toString(), null));
        } else if (methodName.equals(START_WITH_CLASSIFIER)) {
            poseDetector.process(inputImage)
                    .continueWith(
                            classificationExecutor,
                            task -> {
                                Pose pose = task.getResult();
                                List<String> classificationResult = new ArrayList<>();

                                if (poseClassifierProcessor == null) {
                                    poseClassifierProcessor = new PoseClassifierProcessor(context, false);
                                }

                                classificationResult = poseClassifierProcessor.getPoseResult(pose);
                                return new PoseWithClassification(pose, classificationResult);
                            }
                    )
                    .addOnSuccessListener(
                            (OnSuccessListener<PoseWithClassification>) poseWithClassification -> {
                                List<Map<String, Object>> poseList = new ArrayList<>();
                                final Pose pose = poseWithClassification.pose;

                                if (!pose.getAllPoseLandmarks().isEmpty()) {
                                    Map<String, Object> poseMap = new HashMap<String, Object>();
                                    List<Map<String, Object>> landmarks = new ArrayList<>();
                                    for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
                                        Map<String, Object> landmarkMap = new HashMap<>();
                                        landmarkMap.put("type", poseLandmark.getLandmarkType());
                                        landmarkMap.put("x", poseLandmark.getPosition3D().getX());
                                        landmarkMap.put("y", poseLandmark.getPosition3D().getY());
                                        landmarkMap.put("z", poseLandmark.getPosition3D().getZ());
                                        landmarkMap.put("likelihood", poseLandmark.getInFrameLikelihood());
                                        landmarks.add(landmarkMap);
                                    }
                                    poseMap.put("landmarks", landmarks);
                                    poseMap.put("name", PoseDataStorage.getPose());
                                    poseMap.put("accuracy", PoseDataStorage.getAccuracy());
                                    poseList.add(poseMap);
                                }
                                result.success(poseList);
                            }
                    )
                    .addOnFailureListener(e -> result.error("PoseDetectorClassifierError", e.toString(), null));
        } else if(methodName.equals(START_ACTIVITY)) {
            poseDetector.process(inputImage)
                    .continueWith(
                            classificationExecutor,
                            task -> {
                                Pose pose = task.getResult();
                                List<String> classificationResult = new ArrayList<>();

                                if (poseClassifierProcessor == null) {
                                    poseClassifierProcessor = new PoseClassifierProcessor(context, true);
                                }

                                classificationResult = poseClassifierProcessor.getPoseResultWithReps(pose);
                                return new PoseWithClassification(pose, classificationResult);
                            }
                    )
                    .addOnSuccessListener(
                            (OnSuccessListener<PoseWithClassification>) poseWithClassification -> {
                                List<Map<String, Object>> poseList = new ArrayList<>();
                                final Pose pose = poseWithClassification.pose;

                                if (!pose.getAllPoseLandmarks().isEmpty()) {
                                    Map<String, Object> poseMap = new HashMap<String, Object>();
                                    List<Map<String, Object>> landmarks = new ArrayList<>();
                                    for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
                                        Map<String, Object> landmarkMap = new HashMap<>();
                                        landmarkMap.put("type", poseLandmark.getLandmarkType());
                                        landmarkMap.put("x", poseLandmark.getPosition3D().getX());
                                        landmarkMap.put("y", poseLandmark.getPosition3D().getY());
                                        landmarkMap.put("z", poseLandmark.getPosition3D().getZ());
                                        landmarkMap.put("likelihood", poseLandmark.getInFrameLikelihood());
                                        landmarks.add(landmarkMap);
                                    }
                                    poseMap.put("landmarks", landmarks);
                                    poseMap.put("name", PoseDataStorage.getPose());
                                    poseMap.put("accuracy", PoseDataStorage.getAccuracy());
                                    poseMap.put("reps", RepetitionCounter.numRepeats);
                                    poseList.add(poseMap);
                                }
                                result.success(poseList);
                            }
                    )
                    .addOnFailureListener(e -> result.error("PoseDetectorClassifierError", e.toString(), null));
        }

    }

    private void closeDetector() {
        poseDetector.close();
        poseClassifierProcessor = null;
        PoseDataStorage.setData(null, 0.0);
        RepetitionCounter.numRepeats = 0;
    }
}
