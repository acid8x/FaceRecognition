package com.example.facerecognition;

import java.lang.Math;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceRecognizerSF;

import android.app.AlertDialog;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.Toast;

public class MainActivity extends CameraActivity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";

    private static double cosine_similar_threshold = 0.363;
    private static double l2norm_similar_threshold = 1.128;

    private Mat mRgba, mBgr, mBgrScaled;
    private Size mInputSize = null;
    private float mScale = 2.f;
    private boolean isDialogOpen = false;
    private int missedFrame = 0;
    private MatOfByte mModelBufferFD, mConfigBufferFD, mModelBufferFR, mConfigBufferFR;
    private FaceDetectorYN mFaceDetector;
    private FaceRecognizerSF mFaceRecognizerSF;
    private CameraBridgeViewBase mOpenCvCameraView;
    private List<Face> faceList = new ArrayList<>();
    private List<Face> prevRecognizedFaces = new ArrayList<>();

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (!OpenCVLoader.initLocal()) return;
        mFaceDetector = getFaceDetector();
        mFaceRecognizerSF = getFaceRecognizerSF();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) mOpenCvCameraView.enableView();
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mBgr = new Mat();
        mBgrScaled = new Mat();
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mBgr.release();
        mBgrScaled.release();
    }

    public void visualize(Mat rgba, Mat faces) {
        String error = faces.rows() + " faces";
        boolean isFaceRecognized = false;
        List<Face> recognizedFaces = new ArrayList<>();
        float[] faceData = new float[faces.cols() * faces.channels()];
        for (int i = 0; i < faces.rows(); i++) {
            faces.get(i, 0, faceData);
            Rect faceRect = new Rect(Math.round(faceData[0]), Math.round(faceData[1]), Math.round(faceData[2]), Math.round(faceData[3]));
            Rect faceRectScaled = new Rect(Math.round(mScale*faceData[0]), Math.round(mScale*faceData[1]), Math.round(mScale*faceData[2]), Math.round(mScale*faceData[3]));
            if (!isValidRect(faceRect, mBgrScaled)) {
                error += ", face[" + i + "] not valid rect";
                continue;
            }
            Imgproc.rectangle(rgba, faceRectScaled, new Scalar(0, 255, 0), 2);
            String personName = null;
            int qualityScore = FaceQualityEvaluator.evaluateFaceQuality(rgba, faceRectScaled, 0.9);
            double distance = 1000;
            Face recognizedFace = null;
            for (Face f : prevRecognizedFaces) {
                double d = f.getRectDistance(faceRectScaled);
                error += ", face[" + i + "] " + f.getName() + " is at " + d + " width: " + faceRectScaled.width;
                if (d < distance) {
                    recognizedFace = f;
                    distance = d;
                }
            }
            if (recognizedFace != null && distance < 20) {
                prevRecognizedFaces.get(prevRecognizedFaces.indexOf(recognizedFace)).setRect(faceRectScaled);
                personName = recognizedFace.getName();
                recognizedFace.setRect(faceRectScaled);
                recognizedFaces.add(recognizedFace);
            }
            if (personName == null && !isDialogOpen && qualityScore >= 75) {
                Mat feature = new Mat();
                mFaceRecognizerSF.feature(mBgrScaled, feature);
                feature = feature.clone();
                double cos = 0;
                double l2n = 0;
                Face recFace = null;
                for (Face f : faceList) {
                    double c = mFaceRecognizerSF.match(feature, f.getFeatureMat(), FaceRecognizerSF.FR_COSINE);
                    double l = mFaceRecognizerSF.match(feature, f.getFeatureMat(), FaceRecognizerSF.FR_NORM_L2);
                    error += ", face[" + i + "] " + f.getName() + " cos: " + c + " l2: " + l;
                    if (c >= cosine_similar_threshold && l <= l2norm_similar_threshold) {
                        if (recFace == null) {
                            recFace = f;
                            cos = c;
                            l2n = l;
                        } else if (c >= cos && l >= l2n) {
                            recFace = f;
                            cos = c;
                            l2n = l;
                        }
                    }
                }
                if (recFace != null) {
                    personName = recFace.getName();
                    recFace.setRect(faceRectScaled);
                    recognizedFaces.add(recFace);
                } else if (missedFrame > 10) askForName(feature, faceRectScaled);
            }
            if (personName == null) personName = "Inconnu";
            else isFaceRecognized = true;
            error += ": Detected name " + personName;
            Log.i("MYINFO", error);
            Imgproc.putText(rgba, personName + " Q" + qualityScore,
                    new Point(faceRectScaled.x, faceRectScaled.y + faceRectScaled.height + 20),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);
        }
        if (!isDialogOpen) {
            if (!isFaceRecognized) missedFrame++;
            else missedFrame = 0;
            if (!recognizedFaces.isEmpty() || missedFrame > 10) prevRecognizedFaces = recognizedFaces;
        }
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        if (mFaceDetector != null) {
            Size inputSize = new Size(Math.round(mRgba.cols() / mScale), Math.round(mRgba.rows() / mScale));
            if (mInputSize == null || !mInputSize.equals(inputSize)) {
                mInputSize = inputSize;
                mFaceDetector.setInputSize(mInputSize);
            }
            Imgproc.cvtColor(mRgba, mBgr, Imgproc.COLOR_RGBA2BGR);
            Imgproc.resize(mBgr, mBgrScaled, mInputSize);
            Mat mFaces = new Mat();
            int status = mFaceDetector.detect(mBgrScaled, mFaces);
            Log.d(TAG, "Detector returned status " + status);
            visualize(mRgba, mFaces);
        }
        return mRgba;
    }

    private void askForName(Mat feature, Rect rect) {
        isDialogOpen = true;
        runOnUiThread(() -> {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("Nouvelle personne détectée");
            final EditText input = new EditText(this);
            builder.setView(input);
            builder.setPositiveButton("OK", (dialog, which) -> {
                String name = input.getText().toString();
                if (!name.isEmpty()) {
                    faceList.add(new Face(name,feature,rect));
                }
                missedFrame = 0;
                isDialogOpen = false;
            });
            builder.setNegativeButton("Annuler", (dialog, which) -> {
                missedFrame = 0;
                isDialogOpen = false;
                dialog.cancel();
            });
            builder.show();
        });
    }

    private boolean isValidRect(Rect rect, Mat mat) {
        return rect.x >= 0 && rect.y >= 0 && rect.width > 0 && rect.height > 0 && rect.x + rect.width <= mat.cols() && rect.y + rect.height <= mat.rows();
    }

    private FaceDetectorYN getFaceDetector() {
        byte[] buffer;
        try {
            InputStream is = getResources().openRawResource(R.raw.face_detection_yunet_2023mar);
            int size = is.available();
            buffer = new byte[size];
            int bytesRead = is.read(buffer);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        mModelBufferFD = new MatOfByte(buffer);
        mConfigBufferFD = new MatOfByte();

        return FaceDetectorYN.create("onnx", mModelBufferFD, mConfigBufferFD, new Size(320, 320));
    }

    private FaceRecognizerSF getFaceRecognizerSF() {
        byte[] buffer;
        try {
            InputStream is = getResources().openRawResource(R.raw.face_recognition_sface_2021dec);
            int size = is.available();
            buffer = new byte[size];
            int bytesRead = is.read(buffer);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        mModelBufferFR = new MatOfByte(buffer);
        mConfigBufferFR = new MatOfByte();

        return FaceRecognizerSF.create("onnx", mModelBufferFR, mConfigBufferFR);
    }
}