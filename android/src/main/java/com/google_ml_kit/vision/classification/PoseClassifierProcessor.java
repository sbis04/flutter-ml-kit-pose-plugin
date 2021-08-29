package com.google_ml_kit.vision.classification;

import android.content.Context;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.WorkerThread;

import com.google.common.base.Preconditions;
import com.google.mlkit.vision.pose.Pose;
import com.google_ml_kit.vision.PoseDataStorage;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Accepts a stream of {@link Pose} for classification
 */
public class PoseClassifierProcessor {
    private static final String TAG = "PoseClassifierProcessor";
    private static final String POSE_SAMPLES_FILE = "pose/fitness_poses_csvs_out_all.csv";

    final private EMASmoothing emaSmoothing;
    private PoseClassifier poseClassifier;

    @WorkerThread
    public PoseClassifierProcessor(Context context) {
        Preconditions.checkState(Looper.myLooper() != Looper.getMainLooper());
        emaSmoothing = new EMASmoothing();

        loadPoseSamples(context);
    }

    private void loadPoseSamples(Context context) {
        List<PoseSample> poseSamples = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(context.getAssets().open(POSE_SAMPLES_FILE)));
            String csvLine = reader.readLine();
            while (csvLine != null) {
                // If line is not a valid {@link PoseSample}, we'll get null and skip adding to the list.
                PoseSample poseSample = PoseSample.getPoseSample(csvLine, ",");
                if (poseSample != null) {
                    poseSamples.add(poseSample);
                }
                csvLine = reader.readLine();
            }
        } catch (IOException e) {
            Log.e(TAG, "Error when loading pose samples.\n" + e);
        }
        poseClassifier = new PoseClassifier(poseSamples);
    }

    /**
     * Given a new {@link Pose} input, returns a list of formatted {@link String}s with Pose
     * classification results.
     *
     * <p>Currently it returns up to 2 strings as following:
     * 1: PoseClass : [0.0-1.0] confidence
     */
    @WorkerThread
    public List<String> getPoseResult(Pose pose) {
        Preconditions.checkState(Looper.myLooper() != Looper.getMainLooper());
        List<String> result = new ArrayList<>();

        ClassificationResult classification = poseClassifier.classify(pose);
        classification = emaSmoothing.getSmoothedResult(classification);

        if (pose.getAllPoseLandmarks().isEmpty()) {
            return result;
        }

        // Add maxConfidence class of current frame to result if pose is found.
        if (!pose.getAllPoseLandmarks().isEmpty()) {
            String maxConfidenceClass = classification.getMaxConfidenceClass();
            float poseAccuracy = classification.getClassConfidence(maxConfidenceClass) / poseClassifier.confidenceRange();

            // THESE ARE REQUIRED IN FLUTTER
            // -----------------------------------------------------------------------------------
            // Log.i("POSE DETECTED: ", maxConfidenceClass);
            // Log.i("POSE ACCURACY: ", String.format(Locale.US, "%.2f", poseAccuracy));
            // -----------------------------------------------------------------------------------
            //
            PoseDataStorage.setData(maxConfidenceClass, Double.parseDouble(String.format(Locale.US, "%.2f", poseAccuracy)));
        }

        return result;
    }

}
