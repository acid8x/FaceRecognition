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

    public boolean isLastRecognizedFace(Rect newRect) {
        if (rect == null) {
            return false;
        }
        if (newRect.width - 20 >= rect.width || newRect.width + 20 <= rect.width || newRect.height - 20 >= rect.height || newRect.height + 20 <= rect.height) {
            return false;
        }
        if (newRect.x + 10 >= rect.x && newRect.x - 10 <= rect.x && newRect.y + 10 >= rect.y && newRect.y - 10 <= rect.y) {
            return true;
        }
        return false;
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
