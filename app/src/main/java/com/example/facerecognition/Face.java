package com.example.facerecognition;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;

public class Face {

    private String name;
    private List<Mat> featureMat;
    private Rect rect;

    public Face(String name, Mat featureMat, Rect rect) {
        this.name = name;
        this.featureMat = new ArrayList<>();
        this.featureMat.add(featureMat);
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

    public List<Mat> getFeatureMatList() {
        return featureMat;
    }

    public void setFeatureMatList(List<Mat> featureMat) {
        this.featureMat = featureMat;
    }

    public Mat getFeatureMat(int id) {
        if (this.featureMat.isEmpty()) return null;
        else if (id >= 0 && id < this.featureMat.size()) return this.featureMat.get(id);
        else return null;
    }

    public void setFeatureMat(int id, Mat feature) {
        int size = 0;
        if (!this.featureMat.isEmpty()) size = this.featureMat.size();
        if (id >= 0 && id < size) this.featureMat.set(id, feature);
        else if (id == size) this.featureMat.add(feature);
    }

    public void setFeatureMat(Mat feature) {
        this.featureMat.add(feature);
    }

    public Rect getRect() {
        return rect;
    }

    public void setRect(Rect rect) {
        this.rect = rect;
    }
}
