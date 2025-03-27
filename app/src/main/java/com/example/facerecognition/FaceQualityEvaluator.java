package com.example.facerecognition;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class FaceQualityEvaluator {

    public static int evaluateFaceQuality(Mat image, Rect faceRect, double confidenceScore) {
        int score = 0;

        // 1. **Taille du visage**
        int frameArea = image.width() * image.height();
        int faceArea = faceRect.width * faceRect.height;
        double sizeRatio = (double) faceArea / frameArea;
        int sizeScore = (int) (sizeRatio * 300); // Normalisé sur 30 points
        sizeScore = Math.min(sizeScore, 30);
        score += sizeScore;

        // 2. **Angle de la tête (simple estimation)**
        double anglePenalty = Math.abs(estimateHeadAngle(faceRect));
        int angleScore = (int) (20 - anglePenalty * 2);
        angleScore = Math.max(angleScore, 0);
        score += angleScore;

        // 3. **Flou (Variance de Laplacien)**
        int blurScore = evaluateSharpness(image);
        blurScore = Math.min(blurScore, 20);
        score += blurScore;

        // 4. **Luminosité**
        int brightnessScore = evaluateBrightness(image);
        brightnessScore = Math.min(brightnessScore, 20);
        score += brightnessScore;

        // 5. **Score de confiance (si disponible)**
        int confidenceScaled = (int) (confidenceScore * 10);
        score += Math.min(confidenceScaled, 10);

        return Math.min(score, 100);
    }

    // **Estime si la tête est bien droite**
    private static double estimateHeadAngle(Rect faceRect) {
        return Math.abs((double) faceRect.width / faceRect.height - 1) * 20;
    }

    // **Détection du flou avec Variance de Laplacien**
    private static int evaluateSharpness(Mat image) {
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        Core.meanStdDev(laplacian, mean, std);
        double variance = Math.pow(std.get(0, 0)[0], 2);
        return (int) Math.min(variance / 10, 20);
    }

    // **Évalue la luminosité**
    private static int evaluateBrightness(Mat image) {
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        Scalar meanIntensity = Core.mean(gray);
        double brightness = meanIntensity.val[0];
        return (int) (20 - Math.abs(128 - brightness) / 6);
    }

    // **Affiche le score sous le visage**
    public static void drawScoreOnFace(Mat image, Rect faceRect, int score) {
        Point textPosition = new Point(faceRect.x, faceRect.y + faceRect.height + 30);
        Imgproc.putText(image, "Score: " + score + "/100", textPosition, Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
    }
}