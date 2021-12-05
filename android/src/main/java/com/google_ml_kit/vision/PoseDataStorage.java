package com.google_ml_kit.vision;

public class PoseDataStorage {
    static private String pose;
    static private double accuracy;
    static private int reps;

    // Getters
    static public double getAccuracy() {
        return accuracy;
    }

    static public String getPose() {
        return pose;
    }

    static public int getNumReps() {
        return reps;
    }

    // Setter
    static public void setData(String newPose, double newAccuracy) {
        pose = newPose;
        accuracy = newAccuracy;
    }
}
