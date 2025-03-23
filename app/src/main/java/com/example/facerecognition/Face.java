package com.example.facerecognition;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

public class Face {

    private String name;
    private Mat mat;
    private Mat featureMat;
    private Rect rect;

    public Face(String name, Mat mat, Mat featureMat, Rect rect) {
        this.name = name;
        this.mat = mat;
        this.featureMat = featureMat;
        this.rect = rect;
    }

    public boolean isLastRecognizedFace(Rect newRect) {
        if (rect == null) return false;
        else if (rect.x - 5 >= newRect.x || rect.x + 5 <= newRect.x) return false;
        else if (rect.y - 5 >= newRect.y || rect.y + 5 <= newRect.y) return false;
        else if (rect.width - 10 >= newRect.width || rect.width + 10 <= newRect.width) return false;
        else return rect.height - 10 < newRect.height && rect.height + 10 > newRect.height;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Mat getMat() {
        return mat;
    }

    public void setMat(Mat mat) {
        this.mat = mat;
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
