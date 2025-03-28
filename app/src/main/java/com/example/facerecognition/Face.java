package com.example.facerecognition;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

public class Face {

    private String name;
    private Mat featureMat;
    private Rect rect;

    public Face(String name, Mat featureMat, Rect rect) {
        this.name = name;
        this.featureMat = featureMat;
        this.rect = rect;
    }

    public double getRectDistance(Rect newRect) {
        double d = Math.sqrt((newRect.y - rect.y) * (newRect.y - rect.y) + (newRect.x - rect.x) * (newRect.x - rect.x));
        d /= ((double) newRect.width / 100);
        return d;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Mat getFeatureMat() {
        return featureMat;
    }

    public void setFeatureMat(Mat featureMat) {
        this.featureMat = featureMat;
    }

    public Rect getRect() {
        return rect;
    }

    public void setRect(Rect rect) {
        this.rect = rect;
    }
}
