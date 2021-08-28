package com.google_ml_kit.vision;

import android.content.Context;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseLandmark;
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;
import com.google_ml_kit.ApiDetectorInterface;
import com.google_ml_kit.vision.classification.PoseClassifierProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;

//Detector to the pose landmarks present in a image.
//Creates an abstraction over PoseDetector provided by ml kit.
public class PoseDetector implements ApiDetectorInterface {
    private static final String START = "vision#startPoseDetector";
    private static final String CLOSE = "vision#closePoseDetector";

    private final Context context;
    private com.google.mlkit.vision.pose.PoseDetector poseDetector;
    private PoseClassifierProcessor poseClassifierProcessor;
    private final Executor classificationExecutor;
    private int count = 0;

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
                Arrays.asList(START,
                        CLOSE));
    }

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull MethodChannel.Result result) {
        String method = call.method;
        if (method.equals(START)) {
            handleDetection(call, result);
        } else if (method.equals(CLOSE)) {
            closeDetector();
            result.success(null);
        } else {
            result.notImplemented();
        }
    }

    private void handleDetection(MethodCall call, final MethodChannel.Result result) {
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
        int detectorMode = PoseDetectorOptions.STREAM_MODE;
        if (mode.equals("single")) {
            detectorMode = PoseDetectorOptions.SINGLE_IMAGE_MODE;
        }
        if (model.equals("base")) {
            PoseDetectorOptions detectorOptions = new PoseDetectorOptions.Builder()
                    .setDetectorMode(detectorMode)
                    .build();
            poseDetector = PoseDetection.getClient(detectorOptions);
        } else {
            AccuratePoseDetectorOptions detectorOptions = new AccuratePoseDetectorOptions.Builder()
                    .setDetectorMode(detectorMode)
                    .build();
            poseDetector = PoseDetection.getClient(detectorOptions);
        }

        poseDetector.process(inputImage)
//                .continueWith(
//                        classificationExecutor,
//                        task -> {
//                            Pose pose = task.getResult();
//                            List<String> classificationResult = new ArrayList<>();
//
//                            if (poseClassifierProcessor == null) {
//                                poseClassifierProcessor = new PoseClassifierProcessor(context, true);
//                            }
//                            classificationResult = poseClassifierProcessor.getPoseResult(pose);
//                            Log.i("POSE CLASSIFIER", classificationResult.get(0));
//                            return new PoseWithClassification(pose, classificationResult);
//                        }
//                )
                .addOnSuccessListener(
//                        classificationExecutor,
                        (OnSuccessListener<Pose>) pose -> {
                            List<List<Map<String, Object>>> array = new ArrayList<>();
                            if (!pose.getAllPoseLandmarks().isEmpty()) {
                                List<Map<String, Object>> landmarks = new ArrayList<>();
                                for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
                                    Map<String, Object> landmarkMap = new HashMap<>();
                                    landmarkMap.put("type", poseLandmark.getLandmarkType());
                                    landmarkMap.put("x", poseLandmark.getPosition3D().getX());
                                    landmarkMap.put("y", poseLandmark.getPosition3D().getY());
                                    landmarkMap.put("z", poseLandmark.getPosition3D().getZ());
                                    landmarkMap.put("likelihood", poseLandmark.getInFrameLikelihood());
                                    landmarkMap.put("pose", PoseDataStorage.getPose());
                                    landmarkMap.put("accuracy", PoseDataStorage.getAccuracy());
                                    landmarks.add(landmarkMap);
                                }
//                                List<String> classificationResult = new ArrayList<>();
//
//                                if (poseClassifierProcessor == null) {
//                                    poseClassifierProcessor = new PoseClassifierProcessor(context, true);
//                                }
//                                classificationResult = poseClassifierProcessor.getPoseResult(pose);
//                                Log.i("POSE CLASSIFIER", classificationResult.get(0));
                                array.add(landmarks);
                            }
                            result.success(array);
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                result.error("PoseDetectorError", e.toString(), null);
                            }
                        })
                .continueWith(
                        classificationExecutor,
                        task -> {
                            Pose pose = task.getResult();
                            List<String> classificationResult = new ArrayList<>();

                            if (poseClassifierProcessor == null) {
                                poseClassifierProcessor = new PoseClassifierProcessor(context, true);
                            }

                            classificationResult = poseClassifierProcessor.getPoseResult(pose);
                            return new PoseWithClassification(pose, classificationResult);
                        }
                );

    }

    private void closeDetector() {
        poseDetector.close();
    }
}