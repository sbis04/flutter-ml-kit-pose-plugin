package com.google_ml_kit.vision.classification;

import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * Runs EMA smoothing over a window with given stream of pose classification results.
 */
public class EMASmoothing {
  private static final int DEFAULT_WINDOW_SIZE = 10;
  private static final float DEFAULT_ALPHA = 0.2f;

  private final int windowSize;
  private final float alpha;
  // This is a window of {@link ClassificationResult}s as outputted by the {@link PoseClassifier}.
  // We run smoothing over this window of size {@link windowSize}.
  private final Deque<ClassificationResult> window;

  public EMASmoothing() {
    this(DEFAULT_WINDOW_SIZE, DEFAULT_ALPHA);
  }

  public EMASmoothing(int windowSize, float alpha) {
    this.windowSize = windowSize;
    this.alpha = alpha;
    this.window = new LinkedBlockingDeque<>(windowSize);
  }

  public ClassificationResult getSmoothedResult(ClassificationResult classificationResult) {
    // If we are at window size, remove the last (oldest) result.
    if (window.size() == windowSize) {
      window.pollLast();
    }
    // Insert at the beginning of the window.
    window.addFirst(classificationResult);

    Set<String> allClasses = new HashSet<>();
    for (ClassificationResult result : window) {
      allClasses.addAll(result.getAllClasses());
    }

    ClassificationResult smoothedResult = new ClassificationResult();

    for (String className : allClasses) {
      float factor = 1;
      float topSum = 0;
      float bottomSum = 0;
      for (ClassificationResult result : window) {
        float value = result.getClassConfidence(className);

        topSum += factor * value;
        bottomSum += factor;

        factor = (float) (factor * (1.0 - alpha));
      }
      smoothedResult.putClassConfidence(className, topSum / bottomSum);
    }

    return smoothedResult;
  }
}
