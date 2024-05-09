/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.agenew.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;

import com.mediatek.neuropilot_S.Interpreter;
import com.mediatek.neuropilot_S.Tensor;
import com.mediatek.neuropilot_S.nnapi.NnApiDelegate;
//import com.mediatek.neuropilot.Interpreter.Options;
//import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.Tensor;
import com.agenew.detection.MainActivity;
import com.agenew.detection.env.Logger;
import com.agenew.detection.env.Utils;

import org.tensorflow.lite.gpu.GpuDelegate;
//import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;


/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 * <p>
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class YoloV8Classifier implements Classifier {
    private static final String TAG = "YoloV8Classifier";

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param isQuantized   Boolean representing model is quantized or not
     */
    public static YoloV8Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final boolean isQuantized,
            final int inputSize)
            throws IOException {
        final YoloV8Classifier d = new YoloV8Classifier();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();
        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
//        try {
//            Interpreter.Options options = (new Interpreter.Options());
//            options.setNumThreads(NUM_THREADS);
//            if (isNNAPI) {
//                d.nnapiDelegate = null;
//                // Initialize interpreter with NNAPI delegate for Android Pie or above
//                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//                    d.nnapiDelegate = new NnApiDelegate();
//                    options.addDelegate(d.nnapiDelegate);
//                    options.setNumThreads(NUM_THREADS);
//                    options.setUseNNAPI(false);
//                    options.setAllowFp16PrecisionForFp32(true);
//                    options.setAllowBufferHandleOutput(true);
//                    options.setUseNNAPI(true);
//                }
//            }
//            if (isGPU) {
//                GpuDelegate.Options gpu_options = new GpuDelegate.Options();
//                gpu_options.setPrecisionLossAllowed(true); // It seems that the default is true
//                gpu_options.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
//                d.gpuDelegate = new GpuDelegate(gpu_options);
////                options.addDelegate(d.gpuDelegate);
//            }
//            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
//            interpreter = new Interpreter(loadModelFile(activity, MODEL_PATH));
////            d.tfLite = new Interpreter(d.tfliteModel, options);
//            Log.d("lcq", "interpreter: " + d.tfLite.toString());
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.INPUT_SIZE = inputSize;
        d.imgData = ByteBuffer.allocateDirect(1 * d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel);
        Log.d("lcq", "d.imgData: " + d.imgData);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.INPUT_SIZE * d.INPUT_SIZE];
        Log.d("lcq", "d.intValues: " + d.intValues);

//        d.output_box = (int) ((Math.pow((inputSize / 32), 2) + Math.pow((inputSize / 16), 2) + Math.pow((inputSize / 8), 2)) * 3);
//        Log.d("lcq", "d.output_box: " + d.output_box);
//        d.OUTPUT_WIDTH = output_width;
//        d.MASKS = masks;
//        d.ANCHORS = anchors;
        if (d.isModelQuantized){
            Tensor inpten = d.tfLite.getInputTensor(0);
            d.inp_scale = inpten.quantizationParams().getScale();
            d.inp_zero_point = inpten.quantizationParams().getZeroPoint();
            Tensor oupten = d.tfLite.getOutputTensor(0);
            d.oup_scale = oupten.quantizationParams().getScale();
            d.oup_zero_point = oupten.quantizationParams().getZeroPoint();
        }

        int[] shape = d.tfLite.getOutputTensor(0).shape();
        d.output_box = shape[2];
        Log.d("lcq", "d.output_box: " + d.output_box);
        //int numClass = shape[shape.length - 1] - 4;
        d.numClass = shape[1] - 4;
        Log.d("lcq", "shape[1] : " + shape[1]);
        //d.outData = ByteBuffer.allocateDirect(8300);
        d.outData = ByteBuffer.allocateDirect(d.output_box * (d.numClass + 4) * numBytesPerChannel);
        d.outData.order(ByteOrder.nativeOrder());
        return d;
    }

    public int getInputSize() {
        return INPUT_SIZE;
    }


    @Override
    public void close() {
        tfLite.close();
        tfLite = null;
//        if (gpuDelegate != null) {
//            gpuDelegate.close();
//            gpuDelegate = null;
//        }
        if (nnapiDelegate != null) {
            nnapiDelegate.close();
            nnapiDelegate = null;
        }
        tfliteModel = null;
    }

    public void setNumThreads(int num_threads) {
        if (tfLite != null)
            tfLite.setNumThreads(num_threads);
    }


    private void recreateInterpreter() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
//        if (gpuDelegate == null) {
//            gpuDelegate = new GpuDelegate();
//            tfliteOptions.addDelegate(gpuDelegate);
//            recreateInterpreter();
//        }
    }

    public void useCPU() {
        recreateInterpreter();
    }

    public void useNNAPI() {
        nnapiDelegate = new NnApiDelegate();
        tfliteOptions.addDelegate(nnapiDelegate);
        recreateInterpreter();
    }

	@Override
	public float getObjThresh() {
		return MainActivity.MINIMUM_CONFIDENCE_TF_OD_API;
	}

    private static final Logger LOGGER = new Logger();

    // Float model
    private final float IMAGE_MEAN = 0;

    private final float IMAGE_STD = 255.0f;

    //config yolo
    private int INPUT_SIZE = -1;

    //    private int[] OUTPUT_WIDTH;
//    private int[][] MASKS;
//    private int[] ANCHORS;
    private int output_box;

	// Number of threads in the java app
	private static final int NUM_THREADS = 1;

	private boolean isModelQuantized;

	/** holds a gpu delegate */
//    GpuDelegate gpuDelegate = null;
	/** holds an nnapi delegate */
	NnApiDelegate nnapiDelegate = null;

    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    // Config values.

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;

    private ByteBuffer imgData;
    private ByteBuffer outData;

    private Interpreter tfLite;
    private float inp_scale;
    private int inp_zero_point;
    private float oup_scale;
    private int oup_zero_point;
    private int numClass;

    private YoloV8Classifier() {
    }

//    public float iou(float[] box1, float[] box2) {
//        float x1 = Math.max(box1[0], box2[0]);
//        float y1 = Math.max(box1[1], box2[1]);
//        float x2 = Math.min(box1[2], box2[2]);
//        float y2 = Math.min(box1[3], box2[3]);
//
//        float interArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
//
//        float box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
//        float box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
//
//        return interArea / (box1Area + box2Area - interArea);
//    }

//    public static List<float[]> nonMaxSuppression(List<float[]> boxes, List<Float> scores, float threshold) {
//        List<float[]> result = new ArrayList<>();
//
//        while (!boxes.isEmpty()) {
//            // Find the index of the box with the highest score
//            int bestScoreIdx = scores.indexOf(Collections.max(scores));
//            float[] bestBox = boxes.get(bestScoreIdx);
//
//            // Add the box with the highest score to the result
//            result.add(bestBox);
//
//            // Remove the box with the highest score from our lists
//            boxes.remove(bestScoreIdx);
//            scores.remove(bestScoreIdx);
//
//            // Get rid of boxes with high IoU overlap
//            List<float[]> newBoxes = new ArrayList<>();
//            List<Float> newScores = new ArrayList<>();
//            for (int i = 0; i < boxes.size(); i++) {
//                if (iou(bestBox, boxes.get(i)) < threshold) {
//                    newBoxes.add(boxes.get(i));
//                    newScores.add(scores.get(i));
//                }
//            }
//
//            boxes = newBoxes;
//            scores = newScores;
//        }
//
//        return result;
//    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) ((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) (((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        return imgData;
    }

    public List<Recognition> recognizeImage(Bitmap bitmap) {
//        ByteBuffer byteBuffer_ = convertBitmapToByteBuffer(bitmap);
        Bitmap byteBuffer = resizeBitmap(bitmap, 640);
//        Map<Integer, Object> outputMap = new HashMap<>();

//        float[][][] outbuf = new float[1][output_box][labels.size() + 5];
//        outData.rewind();
//        outputMap.put(0, outData);
        Log.d("YoloV5Classifier", "mObjThresh: " + getObjThresh());
        float[][][][] input_arr = bitmapToFloatArray(byteBuffer);
//        Object[] inputArray = {imgData};
//        Log.d("lcq", "inputArray: " + inputArray);
        float[][][] out = new float[1][numClass + 4][output_box];
        tfLite.run(input_arr, out);
//        Log.d("lcq", "outputMap: " + outputMap);

        //ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
        Log.d("lcq", "byteBuffer: " + byteBuffer);
        // byteBuffer.rewind();

        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Log.d("YoloV5Classifier", "out[0] detect start");
//        for (int j = 0; j < output_box; ++j) {
//            for (int i = 0; i < numClass + 4; ++i) {
//                if (isModelQuantized) {
//                    out[0][i][j] = oup_scale * (((int) byteBuffer.get() & 0xFF) - oup_zero_point);
//                } else {
//                    out[0][i][j] = byteBuffer.getFloat();
//                }
//            }
//            // Denormalize xywh
//            for (int i = 0; i < 4; ++i) {
//                out[0][i][j] *= getInputSize();
//            }
//        }
        Log.d("lcq", "out1:" + out[0][5][1]);
        //nms
//        float[][] matrix_2d = out[0];
//        float[][] outputMatrix = new float[8400][6];
//        for (int i = 0; i < 8400; i++) {
//            for (int j = 0; j < 6; j++) {
//                outputMatrix[i][j] = matrix_2d[j][i];
//            }
//        }
//        float threshold = 0.6f; // 类别准确率筛选
//        float non_max = 0.8f; // nms非极大值抑制
//        ArrayList<float[]> boxes = new ArrayList<>();
//        ArrayList<Float> maxScores = new ArrayList();
//        for (float[] detection : outputMatrix) {
//            // 6位数中的后两位是两类的置信度
//            float[] score = Arrays.copyOfRange(detection, 4, 6);
//            float maxValue = score[0];
//            float maxIndex = 0;
//            for (int i = 1; i < score.length; i++) {
//                if (score[i] > maxValue) { // 找出最大的一项
//                    maxValue = score[i];
//                    maxIndex = i;
//                }
//            }
//            if (maxValue >= threshold) { // 如果置信度超过60%则记录
//                detection[4] = maxIndex;
//                detection[5] = maxValue;
//                boxes.add(detection); // 筛选后的框
//                maxScores.add(maxValue); // 筛选后的准确率
//            }
////            List<float[]> results = NonMaxSuppression.nonMaxSuppression(boxes, maxScores, non_max);
////
////            String strResNum = "";
////            String[] names = new String[]{"1", "2"};
////            for (int i = 0; i < results.size(); i++) {
////                // Log.d(TAG,"i:"+i+", result: "+Arrays.toString(results.get(i)));
////                float id = results.get(i)[4];
////                strResNum = strResNum + names[(int) id];
////            }
////            long startTime = System.currentTimeMillis();
////            long endTime = System.currentTimeMillis();
////            long timeElapsed = endTime - startTime;
////            //Log.d(TAG, "strResNum:"+strResNum+", Execution time: " + timeElapsed);
////        }
////        return strResNum;
////    }
//        final float xPos = detection[0];
//        final float yPos = detection[1];
//        final float w = detection[2];
//        final float h = detection[3];
//        final RectF rect =
//                new RectF(
//                        Math.max(0, xPos - w / 2),
//                        Math.max(0, yPos - h / 2),
//                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
//                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
//        detections.add(new Recognition("" + 0, labels.get(),
//                maxScores, rect, detectedClass));
//        Log.d("lcq", "labels.get(detectedClass):" + labels.get(detectedClass));
////            Log.d("lcq", "detections:" + detections);
//    }
//
//    //        Log.d("YoloV5Classifier", "detect end");
//    final ArrayList<Recognition> recognitions = nms(detections);
////        final ArrayList<Recognition> recognitions = detections;
//                return recognitions;
//}


        float[][] matrix_2d = out[0];
        float[][] outputMatrix = new float[output_box][numClass + 4];
        for (int i = 0; i < output_box; i++) {
            for (int j = 0; j < numClass + 4; j++) {
                outputMatrix[i][j] = matrix_2d[j][i];
            }
        }
        for (int j = 0; j < outputMatrix.length; j++) {
            float[] detection = outputMatrix[j];
            int detectedClass = -1;
            float maxClass = 0;
            final int offset = 0;
            float[] score = Arrays.copyOfRange(detection, 4, numClass + 4);
            for (int i = 0; i < score.length; i++) {
                if (score[i] > maxClass) {
                    maxClass = score[i];
                    detectedClass = i;
                }
                //Log.d("lcq","maxClass:"+detectedClass);
            }
            Log.d("lcq", "outputmatrix:" + outputMatrix[j][0]);
            if (maxClass > 0.3F) {
                final float x = outputMatrix[j][0] * 640;
                final float y = outputMatrix[j][1] * 640;
                final float w = outputMatrix[j][2] * 640;
                final float h = outputMatrix[j][3] * 640;
                final RectF rect =
                        new RectF(
                                Math.max(0, x - w / 2),
                                Math.max(0, y - h / 2),
                                Math.min(bitmap.getWidth() - 1, x + w / 2),
                                Math.min(bitmap.getHeight() - 1, y + h / 2));
//                                x-w/2,
//                                y-h/2,
//                                x+w/2,
//                                y+h/2);
                detections.add(new Recognition("" + offset, labels.get(detectedClass),
                        maxClass, rect, detectedClass));
                Log.d("lcq", "detections:" + detections);
                Log.d("lcq", "detectClass:" + detectedClass);
            }
        }
        final ArrayList<Recognition> recognitions = nms(detections);
        return recognitions;
    }
//        for (int j = 0; j < output_box; ++j) {
//            final int offset = 0;
////            final float Score1 = out[0][5][j];
////            final float Score2 = out[0][6][j];
////            final float confidence = Math.max(Score1,Score2);
//            int detectedClass = -1;
//            float maxClass = 0;
//            final float[] classes = new float[labels.size()];
////            Log.d("lcq", "labels.size():" + labels.size());
//            for (int c = 4; c < labels.size(); ++c) {
//                classes[c] = out[0][4 + c][j];
//            }
//            Log.d("lcq", "classes[0]" + classes[0]);
//            for (int c = 0; c < labels.size(); ++c) {
//                if (classes[c] > maxClass) {
//                    detectedClass = c;
//                    maxClass = classes[c];
//                }
//
//            }

//            final float confidenceInClass = maxClass/2;
//            Log.d("lcq", "confidenceInClass:" + confidenceInClass);
//            if (confidenceInClass > 0.5) {
//                final float xPos = out[0][0][j];
//                final float yPos = out[0][1][j];
//
//                final float w = out[0][2][j];
//                final float h = out[0][3][j];
//                Log.d("YoloV5Classifier1",
//                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);
//
//                final RectF rect =
//                        new RectF(
//                                Math.max(0, xPos - w / 2),
//                                Math.max(0, yPos - h / 2),
//                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
//                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
//                detections.add(new Recognition("" + offset, labels.get(detectedClass),
//                        confidenceInClass, rect, detectedClass));
//                Log.d("lcq", "labels.get(detectedClass):" + labels.get(detectedClass));
//            }
////            Log.d("lcq", "detections:" + detections);
//        }
////        Log.d("YoloV5Classifier", "detect end");
//        final ArrayList<Recognition> recognitions = nms(detections);
//////        final ArrayList<Recognition> recognitions = detections;
//        return recognitions;

//        public static float iou(float[] box1, float[] box2) {
//            float x1 = Math.max(box1[0], box2[0]);
//            float y1 = Math.max(box1[1], box2[1]);
//            float x2 = Math.min(box1[2], box2[2]);
//            float y2 = Math.min(box1[3], box2[3]);
//            float interArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
//            float box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
//            float box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
//            return interArea / (box1Area + box2Area - interArea);
//        }

    public boolean checkInvalidateBox(float x, float y, float width, float height, float oriW, float oriH, int intputSize) {
        // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        float halfHeight = height / 2.0f;
        float halfWidth = width / 2.0f;

        float[] pred_coor = new float[]{x - halfWidth, y - halfHeight, x + halfWidth, y + halfHeight};

        // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        float resize_ratioW = 1.0f * intputSize / oriW;
        float resize_ratioH = 1.0f * intputSize / oriH;

        float resize_ratio = resize_ratioW > resize_ratioH ? resize_ratioH : resize_ratioW; //min

        float dw = (intputSize - resize_ratio * oriW) / 2;
        float dh = (intputSize - resize_ratio * oriH) / 2;

        pred_coor[0] = 1.0f * (pred_coor[0] - dw) / resize_ratio;
        pred_coor[2] = 1.0f * (pred_coor[2] - dw) / resize_ratio;

        pred_coor[1] = 1.0f * (pred_coor[1] - dh) / resize_ratio;
        pred_coor[3] = 1.0f * (pred_coor[3] - dh) / resize_ratio;

        // (3) clip some boxes those are out of range
        pred_coor[0] = pred_coor[0] > 0 ? pred_coor[0] : 0;
        pred_coor[1] = pred_coor[1] > 0 ? pred_coor[1] : 0;

        pred_coor[2] = pred_coor[2] < (oriW - 1) ? pred_coor[2] : (oriW - 1);
        pred_coor[3] = pred_coor[3] < (oriH - 1) ? pred_coor[3] : (oriH - 1);

        if ((pred_coor[0] > pred_coor[2]) || (pred_coor[1] > pred_coor[3])) {
            pred_coor[0] = 0;
            pred_coor[1] = 0;
            pred_coor[2] = 0;
            pred_coor[3] = 0;
        }

        // (4) discard some invalid boxes
        float temp1 = pred_coor[2] - pred_coor[0];
        float temp2 = pred_coor[3] - pred_coor[1];
        float temp = temp1 * temp2;
        if (temp < 0) {
            Log.e("checkInvalidateBox", "temp < 0");
            return false;
        }
        if (Math.sqrt(temp) > Float.MAX_VALUE) {
            Log.e("checkInvalidateBox", "temp max");
            return false;
        }

        return true;
    }

    public static Bitmap resizeBitmap(Bitmap source, int maxSize) {
        int outWidth;
        int outHeight;
        int inWidth = source.getWidth();
        int inHeight = source.getHeight();
        if (inWidth > inHeight) {
            outWidth = maxSize;
            outHeight = (inHeight * maxSize) / inWidth;
        } else {
            outHeight = maxSize;
            outWidth = (inWidth * maxSize) / inHeight;
        }

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(source, outWidth, outHeight, false);

        Bitmap outputImage = Bitmap.createBitmap(maxSize, maxSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(outputImage);
        canvas.drawColor(Color.WHITE);
        int left = (maxSize - outWidth) / 2;
        int top = (maxSize - outHeight) / 2;
        canvas.drawBitmap(resizedBitmap, left, top, null);

        return outputImage;
    }

    public static float[][][][] bitmapToFloatArray(Bitmap bitmap) {

        int height = bitmap.getHeight();
        int width = bitmap.getWidth();

        // 初始化一个float数组
        float[][][][] result = new float[1][height][width][3];

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                // 获取像素值
                int pixel = bitmap.getPixel(j, i);
//                if (isModelQuantized) {
//                    // Quantized model
//                    result[0][i][j][0] = (((pixel >> 16) & 0xFF) / 255.0f / );
//                    result[0][i][j][1] = (((pixel >> 8) & 0xFF) / 255.0f / inp_scale );
//                    result[0][i][j][2] = ((pixel & 0xFF) / 255.0f / inp_scale + inp_zero_point);
//                }
                // 将RGB值分离并进行标准化（假设你需要将颜色值标准化到0-1之间）
                result[0][i][j][0] = ((pixel >> 16) & 0xFF) / 255.0f;
                result[0][i][j][1] = ((pixel >> 8) & 0xFF) / 255.0f;
                result[0][i][j][2] = (pixel & 0xFF) / 255.0f;
            }
        }
        return result;
    }


}
