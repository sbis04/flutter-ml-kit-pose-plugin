package com.google_ml_kit.vision;

public class PoseDataStorage {
    static private String pose;
    static private double accuracy;

    // Getters
    static public double getAccuracy() {
        return accuracy;
    }

    static public String getPose() {
        return pose;
    }

    // Setter
    static public void setData(String newPose, double newAccuracy) {
        pose = newPose;
        accuracy = newAccuracy;
    }
}
